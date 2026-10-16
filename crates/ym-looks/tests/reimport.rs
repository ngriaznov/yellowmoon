//! Importing again never duplicates a look (`docs/08-looks.md` sections One look, one entry and
//! Import protocol).
//!
//! The libraries here are made up, laid out like ARRI's five real downloads (a flat folder of
//! `<code> <title>.cube`, group folders of `LL-<code>-<Title>.alf4` and `.aml`, the Mac's junk
//! beside them); only ARRI-like **names** are used, every LUT is our own synthetic one. The owner's
//! realistic sequence of imports runs against one library and the entries are counted after each
//! step: no look twice, no `-2`, nothing written by a step that adds nothing.

mod support;

use std::fs;
use std::path::{Path, PathBuf};

use support::{
    ALF_LUT_SIZE, ALF2_ROOT, ALF2_ZIP, ALF4_ROOT, ALF4_ZIP, alf4_bytes, aml_bytes, cube_text,
    no_progress, open_library, small_db, tree, write_zip,
};
use ym_color::look::LookInput;
use ym_looks::import::{ImportOptions, ImportReport, import_path};
use ym_looks::library::{Library, LookFormat, LookGroup, LookId};
use ym_looks::manifest::sha256_hex;
use ym_looks::recognize::{Fingerprint, PackageDb};

/// Look codes and titles in the shape of ARRI's Look Library. Two looks share a title, as ARRI's do.
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

/// The looks in a library of [`CODED`].
const N: usize = CODED.len();

type Entries = Vec<(String, Vec<u8>)>;

fn group_name(code: u32) -> &'static str {
    match code / 1000 {
        1 => "Black and White",
        2 => "Application",
        3 => "Film",
        4 => "Tinted",
        8 => "Period",
        _ => "Special",
    }
}

fn hyphenated(title: &str) -> String {
    title.replace(" - ", " ").replace(' ', "-")
}

/// The LogC4 3D-LUT zip: one top folder, flat files, no encoding word in the names.
fn logc4_cubes() -> (&'static str, Entries) {
    cube_package(
        "ARRI Look Library LogC4 (log-to-log) 3D-LUTs_data.zip",
        "ARRI Look Library for LogC4 (3D-LUTs)",
        5,
        false,
        0,
    )
}

/// The LogC3 log-to-log 3D-LUT zip.
fn logc3_cubes() -> (&'static str, Entries) {
    cube_package(
        "ARRI Look Library LogC3 (log-to-log) 3D LUTs_data.zip",
        "ARRI Look Library LogC3 (log-to-log)",
        3,
        false,
        100,
    )
}

/// The LogC3 to Rec.709 3D-LUT zip (hyphens in the names).
fn rec709_cubes() -> (&'static str, Entries) {
    cube_package(
        "ARRI Look Library LogC3 (to Rec709) 3D-LUTs_data.zip",
        "ARRI Look Library LogC3 to Rec709 3D-LUTs",
        3,
        true,
        200,
    )
}

fn cube_package(
    zip: &'static str,
    folder: &str,
    size: usize,
    hyphens: bool,
    seed_base: u32,
) -> (&'static str, Entries) {
    let mut entries: Entries = Vec::new();
    for (i, (code, title)) in CODED.iter().enumerate() {
        let file = if hyphens {
            format!("{code}-{}.cube", hyphenated(title))
        } else {
            format!("{code} {title}.cube")
        };
        entries.push((
            format!("{folder}/{file}"),
            cube_text(size, seed_base + i as u32 + 1, "made up").into_bytes(),
        ));
        entries.push((
            format!("__MACOSX/{folder}/._{file}"),
            b"\x00\x05\x16\x07 apple double".to_vec(),
        ));
    }
    (zip, entries)
}

/// The LogC4 look-file zip: group folders of `.alf4` files (their look is the cube's, same name).
fn alf4_files() -> (&'static str, Entries) {
    let entries = CODED
        .iter()
        .enumerate()
        .map(|(i, (code, title))| {
            (
                format!(
                    "{ALF4_ROOT}/{} {}/LL-{code}-{}.alf4",
                    code / 1000 * 1000,
                    group_name(*code),
                    hyphenated(title)
                ),
                alf4_bytes(ALF_LUT_SIZE, i as u32 + 1, None),
            )
        })
        .collect();
    (ALF4_ZIP, entries)
}

/// The LogC3 look-file zip: three strengths of every look.
fn alf2_files() -> (&'static str, Entries) {
    let mut entries = Entries::new();
    for (i, (code, title)) in CODED.iter().enumerate() {
        for (k, strength) in ["Low", "Medium", "High"].into_iter().enumerate() {
            entries.push((
                format!(
                    "{ALF2_ROOT}/{} {}/LL-{code}-{}-{strength}.aml",
                    code / 1000 * 1000,
                    group_name(*code),
                    hyphenated(title)
                ),
                aml_bytes(ALF_LUT_SIZE, 3 * i as u32 + k as u32 + 1, None),
            ));
        }
    }
    (ALF2_ZIP, entries)
}

fn write_tree(root: &Path, entries: &Entries) {
    for (path, bytes) in entries {
        let target = root.join(path);
        fs::create_dir_all(target.parent().unwrap()).unwrap();
        fs::write(&target, bytes).unwrap();
    }
}

fn zip_of(dir: &Path, (name, entries): &(&str, Entries)) -> PathBuf {
    let path = dir.join(name);
    write_zip(&path, entries);
    path
}

/// `db` with the hashes of made-up packages, as `tools/looks/fingerprint.py` records ARRI's: a
/// folder that holds files of several packages is then recognised file by file.
fn fingerprinted(mut db: PackageDb, packages: &[(&str, usize, &Entries)]) -> PackageDb {
    for (id, lut_size, entries) in packages {
        let package = db
            .packages
            .iter_mut()
            .find(|p| p.id == *id)
            .expect("package");
        package.fingerprints = entries
            .iter()
            .filter(|(path, _)| !path.starts_with("__MACOSX"))
            .map(|(path, bytes)| Fingerprint {
                path: path.clone(),
                size: bytes.len() as u64,
                lut_size: *lut_size,
                sha256: sha256_hex(bytes),
            })
            .collect();
    }
    db
}

fn options(input: Option<LookInput>) -> ImportOptions {
    ImportOptions {
        input,
        imported_at: Some("2026-11-02T10:14:00Z".to_string()),
        ..ImportOptions::default()
    }
}

fn import_source(
    library: &mut Library,
    db: &PackageDb,
    path: &Path,
    input: Option<LookInput>,
) -> ImportReport {
    import_path(library, db, path, &options(input), &mut no_progress())
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .1
}

fn manifest_bytes(library: &Library) -> Vec<u8> {
    fs::read(library.paths().manifest_path()).unwrap_or_default()
}

/// What the library holds, as a step leaves it.
#[derive(Debug, PartialEq, Eq)]
struct Counts {
    arri: usize,
    user: usize,
    display: usize,
}

fn counts(library: &Library) -> Counts {
    let mut c = Counts {
        arri: 0,
        user: 0,
        display: library.display_luts().len(),
    };
    for look in library.looks() {
        match look.id.prefix() {
            "arri" => c.arri += 1,
            "user" => c.user += 1,
            _ => {}
        }
    }
    c
}

/// After every step: the entries add up, no look carries a collision number, every entry has its
/// file and every file its entry, and the list reads and verifies clean.
fn check_library(library: &Library, expected: &Counts, label: &str) {
    assert_eq!(&counts(library), expected, "{label}");
    for look in library.looks() {
        let id = look.id.to_string();
        let last = id.rsplit('-').next().unwrap_or("");
        assert!(
            !last.parse::<u32>().is_ok_and(|n| n < 100),
            "{label}: {id} carries a collision number"
        );
    }
    for display in library.display_luts() {
        let id = display.id.to_string();
        assert!(!id.ends_with("-2"), "{label}: {id}");
    }
    let entries = expected.arri + expected.user + expected.display;
    let files: Vec<String> = tree(&library.looks_dir())
        .into_iter()
        .filter(|f| {
            f.starts_with("arri/") || f.starts_with("user/") || f.starts_with("arri-display/")
        })
        .collect();
    assert_eq!(
        files.len(),
        entries,
        "{label}: a file for every entry and an entry for every file: {files:?}"
    );
    assert!(
        library.problems().is_empty(),
        "{label}: {:?}",
        library.problems()
    );
    let mut copy = open_library(library.root());
    assert!(
        copy.verify(&mut no_progress()).unwrap().is_clean(),
        "{label}"
    );
    assert_eq!(&counts(&copy), expected, "{label}: after reopening");
}

fn ids(library: &Library) -> Vec<String> {
    let mut v: Vec<String> = library
        .looks()
        .iter()
        .filter(|l| l.id.prefix() != "builtin")
        .map(|l| l.id.to_string())
        .collect();
    v.sort();
    v
}

fn note_with(report: &ImportReport, text: &str) -> bool {
    report.notes.iter().any(|n| n.contains(text))
}

#[test]
fn the_owners_sequence_of_imports_never_duplicates_a_look() {
    let work = tempfile::tempdir().unwrap();
    let db = small_db(N);
    let mut library = open_library(&work.path().join("app"));

    let logc4 = logc4_cubes();
    let alf4 = alf4_files();
    let logc3 = logc3_cubes();
    let rec709 = rec709_cubes();
    let alf2 = alf2_files();
    let logc4_zip = zip_of(work.path(), &logc4);
    let alf4_zip = zip_of(work.path(), &alf4);
    let logc3_zip = zip_of(work.path(), &logc3);
    let rec709_zip = zip_of(work.path(), &rec709);
    let alf2_zip = zip_of(work.path(), &alf2);

    let c = |arri, user| Counts {
        arri,
        user,
        display: 0,
    };

    // 1. The LogC4 3D-LUT zip.
    let r = import_source(&mut library, &db, &logc4_zip, None);
    assert_eq!(r.added.len(), N, "{r:?}");
    check_library(&library, &c(N, 0), "1 LogC4 cubes");
    let logc4_ids = ids(&library);
    assert_eq!(logc4_ids.len(), N);
    assert!(
        logc4_ids.contains(&"arri:1000:gray-test-1110".to_string()),
        "{logc4_ids:?}"
    );
    let manifest = manifest_bytes(&library);

    // 2. The same zip again: nothing is added and the list is not even rewritten.
    let r = import_source(&mut library, &db, &logc4_zip, None);
    assert!(r.added.is_empty() && r.updated.is_empty(), "{r:?}");
    assert_eq!(r.unchanged.len(), N);
    check_library(&library, &c(N, 0), "2 LogC4 cubes again");
    assert_eq!(
        manifest_bytes(&library),
        manifest,
        "2: the list is unchanged"
    );

    // 3. The ALF4 zip: the same looks as look files. The 3D LUTs stay.
    let r = import_source(&mut library, &db, &alf4_zip, None);
    assert!(r.added.is_empty() && r.updated.is_empty(), "{r:?}");
    assert_eq!(r.unchanged.len(), N);
    assert!(
        note_with(&r, "already imported from the 3D-LUT version"),
        "{:?}",
        r.notes
    );
    check_library(&library, &c(N, 0), "3 ALF4 zip");
    assert_eq!(ids(&library), logc4_ids, "3: the same looks");
    assert_eq!(
        manifest_bytes(&library),
        manifest,
        "3: the list is unchanged"
    );

    // 4. The ALF4 zip and the LogC4 zip together in one import (a folder that holds both).
    let both = work.path().join("both");
    write_tree(&both, &alf4.1);
    write_tree(&both, &logc4.1);
    let known = fingerprinted(
        small_db(N),
        &[
            ("look-library-logc4", 5, &logc4.1),
            ("look-library-logc4-alf4", ALF_LUT_SIZE, &alf4.1),
        ],
    );
    let r = import_source(&mut library, &known, &both, None);
    assert!(r.added.is_empty() && r.updated.is_empty(), "{r:?}");
    assert!(
        note_with(&r, "came both as 3D LUTs and as look files"),
        "{:?}",
        r.notes
    );
    check_library(&library, &c(N, 0), "4 both together");
    assert_eq!(
        manifest_bytes(&library),
        manifest,
        "4: the list is unchanged"
    );

    // 5. The unzipped LogC4 folder, as Finder makes it.
    let unzipped = work
        .path()
        .join("ARRI Look Library LogC4 (log-to-log) 3D-LUTs_data");
    write_tree(&unzipped, &logc4.1);
    let r = import_source(&mut library, &db, &unzipped, None);
    assert!(r.added.is_empty() && r.updated.is_empty(), "{r:?}");
    check_library(&library, &c(N, 0), "5 unzipped folder");
    assert_eq!(
        manifest_bytes(&library),
        manifest,
        "5: the list is unchanged"
    );

    // 6. The LogC3 library: new looks, another set of ids.
    let r = import_source(&mut library, &db, &logc3_zip, None);
    assert_eq!(r.added.len(), N, "{r:?}");
    check_library(&library, &c(2 * N, 0), "6 LogC3");

    // 7. LogC3 to Rec.709 cubes: again new looks.
    let r = import_source(&mut library, &db, &rec709_zip, None);
    assert_eq!(r.added.len(), N, "{r:?}");
    check_library(&library, &c(3 * N, 0), "7 LogC3 to Rec709");

    // 8. The ALF2 zip: three strengths of every look, none of them the Rec.709 cubes.
    let r = import_source(&mut library, &db, &alf2_zip, None);
    assert_eq!(r.added.len(), 3 * N, "{r:?}");
    check_library(&library, &c(6 * N, 0), "8 ALF2");
    let manifest = manifest_bytes(&library);

    // 9. The ALF2 zip again.
    let r = import_source(&mut library, &db, &alf2_zip, None);
    assert!(r.added.is_empty() && r.updated.is_empty(), "{r:?}");
    assert_eq!(r.unchanged.len(), 3 * N);
    check_library(&library, &c(6 * N, 0), "9 ALF2 again");
    assert_eq!(
        manifest_bytes(&library),
        manifest,
        "9: the list is unchanged"
    );

    // The ARRI libraries again, all of them, in any order: still nothing new.
    for zip in [&rec709_zip, &alf4_zip, &logc3_zip, &logc4_zip, &alf2_zip] {
        let r = import_source(&mut library, &db, zip, None);
        assert!(r.added.is_empty() && r.updated.is_empty(), "{zip:?}: {r:?}");
    }
    check_library(&library, &c(6 * N, 0), "all of them again");
    assert_eq!(manifest_bytes(&library), manifest, "the list is unchanged");

    // 10. One cube of the person's own.
    let mine = work.path().join("mine");
    fs::create_dir_all(&mine).unwrap();
    let own = mine.join("Mine.cube");
    fs::write(&own, cube_text(5, 900, "mine")).unwrap();
    let r = import_source(&mut library, &db, &own, Some(LookInput::LogC4));
    assert_eq!(r.added.len(), 1, "{r:?}");
    assert_eq!(r.added[0].id.to_string(), "user:mine");
    check_library(&library, &c(6 * N, 1), "10 one user cube");
    let manifest = manifest_bytes(&library);

    // 11. The same cube renamed (and once more in a folder of its own): the same LUT, no `Mine-2`.
    let renamed = work.path().join("renamed");
    fs::create_dir_all(&renamed).unwrap();
    let copy = renamed.join("Warm Pull.cube");
    fs::copy(&own, &copy).unwrap();
    let r = import_source(&mut library, &db, &copy, Some(LookInput::LogC4));
    assert!(r.added.is_empty() && r.updated.is_empty(), "{r:?}");
    assert_eq!(r.unchanged.len(), 1);
    assert_eq!(r.unchanged[0].id.to_string(), "user:mine");
    assert_eq!(r.unchanged[0].name, "Mine", "the entry keeps its own name");
    assert!(
        note_with(&r, "already imported under another name"),
        "{:?}",
        r.notes
    );
    check_library(&library, &c(6 * N, 1), "11 renamed user cube");
    assert_eq!(
        manifest_bytes(&library),
        manifest,
        "11: the list is unchanged"
    );
    let r = import_source(&mut library, &db, &renamed, Some(LookInput::LogC4));
    assert!(r.added.is_empty() && r.updated.is_empty(), "{r:?}");
    check_library(&library, &c(6 * N, 1), "11 renamed, as a folder");
    assert_eq!(
        manifest_bytes(&library),
        manifest,
        "11: the list is unchanged"
    );

    // The same bytes with another input are another LUT: it is the person's to say.
    let r = import_source(&mut library, &db, &copy, Some(LookInput::LogC3Ei800));
    assert_eq!(r.added.len(), 1, "{r:?}");
    assert_eq!(r.added[0].id.to_string(), "user:warm-pull");
    check_library(&library, &c(6 * N, 2), "11 another input");
}

#[test]
fn look_files_first_and_then_the_cubes_leave_one_entry_per_look() {
    let work = tempfile::tempdir().unwrap();
    let db = small_db(N);
    let mut library = open_library(&work.path().join("app"));
    let c = |arri| Counts {
        arri,
        user: 0,
        display: 0,
    };

    // 12. The ALF4 zip into an empty library, then the LogC4 cubes: they take the entries over.
    let r = import_source(&mut library, &db, &zip_of(work.path(), &alf4_files()), None);
    assert_eq!(r.added.len(), N, "{r:?}");
    check_library(&library, &c(N), "ALF4 zip");
    assert!(
        library
            .looks()
            .iter()
            .filter(|l| l.id.prefix() == "arri")
            .all(|l| l.lut.as_ref().is_some_and(|u| u.format == LookFormat::Alf4))
    );
    let ids_before = ids(&library);

    let r = import_source(
        &mut library,
        &db,
        &zip_of(work.path(), &logc4_cubes()),
        None,
    );
    assert!(r.added.is_empty(), "{r:?}");
    assert_eq!(r.updated.len(), N);
    assert!(
        note_with(&r, "replaced by the 3D-LUT version"),
        "{:?}",
        r.notes
    );
    check_library(&library, &c(N), "then the cubes");
    assert_eq!(ids(&library), ids_before, "the same ids");
    assert!(
        library
            .looks()
            .iter()
            .filter(|l| l.id.prefix() == "arri")
            .all(|l| l.lut.as_ref().is_some_and(|u| u.format.is_cube()))
    );
    assert!(
        tree(&library.looks_dir())
            .iter()
            .all(|f| !f.ends_with(".alf4")),
        "the look files are gone"
    );
    let manifest = manifest_bytes(&library);
    // And the look files once more change nothing.
    let r = import_source(&mut library, &db, &zip_of(work.path(), &alf4_files()), None);
    assert!(r.added.is_empty() && r.updated.is_empty(), "{r:?}");
    assert_eq!(manifest_bytes(&library), manifest);
}

#[test]
fn a_folder_with_both_libraries_gives_one_entry_per_look_the_cubes_winning() {
    let work = tempfile::tempdir().unwrap();
    let db = small_db(N);
    let both = work.path().join("both");
    // The look files sort before the cubes in this folder.
    let (_, alf4) = alf4_files();
    let (_, cubes) = logc4_cubes();
    assert!(
        alf4[0].0 < cubes[0].0,
        "the look files come first in the list"
    );
    write_tree(&both, &alf4);
    write_tree(&both, &cubes);
    let db = fingerprinted(
        db,
        &[
            ("look-library-logc4", 5, &cubes),
            ("look-library-logc4-alf4", ALF_LUT_SIZE, &alf4),
        ],
    );

    let mut library = open_library(&work.path().join("app"));
    let r = import_source(&mut library, &db, &both, None);
    assert_eq!(r.added.len(), N, "{r:?}");
    assert!(r.updated.is_empty(), "{r:?}");
    assert_eq!(
        r.unchanged.len(),
        N,
        "the look files are reported as already there"
    );
    assert!(
        note_with(
            &r,
            "9 looks came both as 3D LUTs and as look files. The 3D LUTs are kept: they have more points."
        ),
        "{:?}",
        r.notes
    );
    let c = Counts {
        arri: N,
        user: 0,
        display: 0,
    };
    check_library(&library, &c, "a folder with both libraries");
    assert!(
        library
            .looks()
            .iter()
            .filter(|l| l.id.prefix() == "arri")
            .all(|l| l.lut.as_ref().is_some_and(|u| u.format.is_cube())),
        "every look is the 3D LUT"
    );
    assert!(
        tree(&library.looks_dir())
            .iter()
            .all(|f| !f.ends_with(".alf4"))
    );
    let manifest = manifest_bytes(&library);
    let r = import_source(&mut library, &db, &both, None);
    assert!(r.added.is_empty() && r.updated.is_empty(), "{r:?}");
    assert_eq!(manifest_bytes(&library), manifest);
}

#[test]
fn a_package_claims_an_own_lut_that_is_its_file_and_keeps_one_entry() {
    let work = tempfile::tempdir().unwrap();
    let db = small_db(N);
    let mut library = open_library(&work.path().join("app"));
    let (_, cubes) = logc4_cubes();
    let (_, alf4) = alf4_files();

    // Earlier the person imported a file of the LogC4 library on its own, as their own LUT, and
    // a look file of it beside it.
    let mine = work.path().join("mine");
    fs::create_dir_all(&mine).unwrap();
    let warm = mine.join("Warm.cube");
    fs::write(&warm, &cubes[2 * 2].1).unwrap(); // 1120 Gray Test - Night
    let cool = mine.join("Cool.alf4");
    fs::write(&cool, &alf4[3].1).unwrap(); // 2110 Studio Sample
    import_source(&mut library, &db, &warm, Some(LookInput::LogC4));
    import_source(&mut library, &db, &cool, None);
    assert_eq!(
        counts(&library),
        Counts {
            arri: 0,
            user: 2,
            display: 0
        }
    );
    let before = tree(&library.looks_dir());
    assert!(before.contains(&"user/Warm.cube".to_string()), "{before:?}");
    assert!(before.contains(&"user/Cool.alf4".to_string()), "{before:?}");

    // Now the package that holds the cube: its looks are the ARRI looks and the person's entry
    // moved to the ARRI group (its ARRI id, name and file), so no look is listed twice.
    let r = import_source(
        &mut library,
        &db,
        &zip_of(work.path(), &logc4_cubes()),
        None,
    );
    assert_eq!(r.added.len(), N - 1, "{r:?}");
    assert_eq!(r.updated.len(), 1, "{r:?}");
    assert_eq!(r.updated[0].id.to_string(), "arri:1000:gray-test-night");
    assert_eq!(r.updated[0].name, "Gray Test Night");
    assert!(
        note_with(&r, "1 of your own LUTs is a file from ARRI's library"),
        "{:?}",
        r.notes
    );
    // The old id is reported with the new one, so what still names it can follow, and said.
    assert_eq!(
        r.renamed,
        vec![(
            LookId::user("warm"),
            LookId::arri(1000, "gray-test-night")
        )]
    );
    assert!(
        note_with(&r, "user:warm is now arri:1000:gray-test-night."),
        "{:?}",
        r.notes
    );
    // Nothing was replaced by a 3D-LUT version of a look file: no look file was involved.
    assert!(
        !note_with(&r, "replaced by the 3D-LUT version"),
        "{:?}",
        r.notes
    );
    check_library(
        &library,
        &Counts {
            arri: N,
            user: 1,
            display: 0,
        },
        "the cubes claim Warm",
    );
    let claimed = library
        .get(&LookId::arri(1000, "gray-test-night"))
        .expect("the claimed look");
    assert_eq!(claimed.name, "Gray Test Night");
    assert!(matches!(
        &claimed.group,
        LookGroup::Arri { group_no: 1000, .. }
    ));
    assert!(
        library.get(&LookId::user("warm")).is_none(),
        "no second entry"
    );
    let files = tree(&library.looks_dir());
    assert!(!files.contains(&"user/Warm.cube".to_string()), "{files:?}");
    assert!(files.contains(&"user/Cool.alf4".to_string()), "{files:?}");
    let manifest = manifest_bytes(&library);

    // The ALF4 zip holds Cool's bytes, but the cube of that look is the library's entry already:
    // the look file is left out and the person's own entry stays theirs.
    let r = import_source(&mut library, &db, &zip_of(work.path(), &alf4_files()), None);
    assert!(r.added.is_empty() && r.updated.is_empty(), "{r:?}");
    check_library(
        &library,
        &Counts {
            arri: N,
            user: 1,
            display: 0,
        },
        "the look files after the cubes",
    );
    assert_eq!(manifest_bytes(&library), manifest);
}

#[test]
fn an_own_look_file_that_is_one_of_the_alf4_libraries_looks_becomes_that_look() {
    let work = tempfile::tempdir().unwrap();
    let db = small_db(N);
    let mut library = open_library(&work.path().join("app"));
    let (_, alf4) = alf4_files();

    let mine = work.path().join("mine");
    fs::create_dir_all(&mine).unwrap();
    let own = mine.join("Cool.alf4");
    fs::write(&own, &alf4[3].1).unwrap();
    import_source(&mut library, &db, &own, None);
    assert_eq!(
        library.get(&LookId::user("cool")).map(|l| l.name.as_str()),
        Some("Cool")
    );

    let r = import_source(&mut library, &db, &zip_of(work.path(), &alf4_files()), None);
    assert_eq!(r.added.len(), N - 1, "{r:?}");
    assert_eq!(r.updated.len(), 1, "{r:?}");
    check_library(
        &library,
        &Counts {
            arri: N,
            user: 0,
            display: 0,
        },
        "the look files claim Cool",
    );
    let files = tree(&library.looks_dir());
    assert!(!files.contains(&"user/Cool.alf4".to_string()), "{files:?}");
    assert!(
        files.iter().filter(|f| f.ends_with(".alf4")).count() == N,
        "{files:?}"
    );
}

#[test]
fn display_luts_follow_the_same_rules() {
    use support::Kind;
    let work = tempfile::tempdir().unwrap();
    let n = 4;
    let mut db = small_db(n);
    // ARRI's display packages have no LUT size to go by (`arri-packages.json`), and no loose cube
    // is taken for one of them because of its size.
    for package in &mut db.packages {
        if package.id.starts_with("display-luts") {
            package.expected.lut_sizes.clear();
        }
    }

    // ARRI's display package, made up with titles that say nothing about ARRI (a loose copy of
    // one of its files is then the person's own LUT until it is asked).
    let displays: Entries = ["Gamma24_Rec709", "P3_D65", "Rec2020_PQ", "HLG"]
        .iter()
        .enumerate()
        .map(|(i, d)| {
            (
                format!("LogC4_to_{d}.cube"),
                cube_text(5, 50 + i as u32, "made up").into_bytes(),
            )
        })
        .collect();
    let zip = zip_of(
        work.path(),
        &(Kind::DisplayLogC4.zip_name(), displays.clone()),
    );

    // The display package twice: nothing new.
    let mut library = open_library(&work.path().join("app"));
    let r = import_source(&mut library, &db, &zip, None);
    assert_eq!(r.added.len(), n, "{r:?}");
    let shown = counts(&library);
    assert_eq!(shown.display, n);
    let manifest = manifest_bytes(&library);
    let r = import_source(&mut library, &db, &zip, None);
    assert!(r.added.is_empty() && r.updated.is_empty(), "{r:?}");
    assert_eq!(manifest_bytes(&library), manifest);

    // A copy of one of the display LUTs, named as the person likes, is the same LUT.
    let mine = work.path().join("mine");
    fs::create_dir_all(&mine).unwrap();
    let copy = mine.join("Mine.cube");
    fs::write(&copy, &displays[1].1).unwrap();
    let r = import_source(&mut library, &db, &copy, Some(LookInput::LogC4));
    assert!(r.added.is_empty() && r.updated.is_empty(), "{r:?}");
    assert_eq!(r.unchanged.len(), 1);
    assert!(r.unchanged[0].display_lut, "{r:?}");
    assert_eq!(counts(&library).user, 0);
    assert_eq!(manifest_bytes(&library), manifest);

    // The other way round: the person's copy came first, the package claims it.
    let mut library = open_library(&work.path().join("app2"));
    import_source(&mut library, &db, &copy, Some(LookInput::LogC4));
    assert_eq!(counts(&library).user, 1);
    let r = import_source(&mut library, &db, &zip, None);
    assert_eq!(r.added.len(), n - 1, "{r:?}");
    assert_eq!(r.updated.len(), 1, "{r:?}");
    assert!(r.updated[0].display_lut, "{r:?}");
    check_library(
        &library,
        &Counts {
            arri: 0,
            user: 0,
            display: n,
        },
        "the package claims the copy",
    );
    let files = tree(&library.looks_dir());
    assert!(!files.iter().any(|f| f.starts_with("user/")), "{files:?}");
}

#[test]
fn two_arri_files_with_the_same_bytes_stay_two_looks() {
    // ARRI's names decide which looks there are, not the bytes: two looks that happen to hold the
    // same LUT are two entries.
    let work = tempfile::tempdir().unwrap();
    let db = small_db(2);
    let mut library = open_library(&work.path().join("app"));
    let same = cube_text(5, 77, "made up").into_bytes();
    let folder = "ARRI Look Library for LogC4 (3D-LUTs)";
    let entries: Entries = vec![
        (format!("{folder}/1110 Alpha.cube"), same.clone()),
        (format!("{folder}/3110 Beta.cube"), same),
    ];
    let zip = zip_of(
        work.path(),
        &(
            "ARRI Look Library LogC4 (log-to-log) 3D-LUTs_data.zip",
            entries,
        ),
    );
    let r = import_source(&mut library, &db, &zip, None);
    assert_eq!(r.added.len(), 2, "{r:?}");
    check_library(
        &library,
        &Counts {
            arri: 2,
            user: 0,
            display: 0,
        },
        "same bytes, two looks",
    );
}

#[test]
fn a_file_that_is_in_the_library_twice_in_one_import_is_added_once() {
    let work = tempfile::tempdir().unwrap();
    let db = small_db(3);
    let mut library = open_library(&work.path().join("app"));
    let src = work.path().join("src");
    fs::create_dir_all(src.join("a")).unwrap();
    fs::create_dir_all(src.join("b")).unwrap();
    fs::write(src.join("a/Film.cube"), cube_text(5, 5, "x")).unwrap();
    fs::write(src.join("b/Film copy.cube"), cube_text(5, 5, "x")).unwrap();
    fs::write(src.join("b/Other.cube"), cube_text(5, 6, "x")).unwrap();
    let r = import_source(&mut library, &db, &src, Some(LookInput::LogC4));
    assert_eq!(r.added.len(), 2, "{r:?}");
    assert_eq!(r.unchanged.len(), 1, "{r:?}");
    assert!(
        note_with(&r, "already imported under another name"),
        "{:?}",
        r.notes
    );
    check_library(
        &library,
        &Counts {
            arri: 0,
            user: 2,
            display: 0,
        },
        "a copy within one import",
    );
}

#[test]
fn a_look_file_replaced_by_its_cube_and_an_own_lut_claimed_in_one_import_are_counted_apart() {
    let work = tempfile::tempdir().unwrap();
    let db = small_db(N);
    let mut library = open_library(&work.path().join("app"));
    let (_, cubes) = logc4_cubes();
    let (_, alf4) = alf4_files();

    // The look files of all but one look (1120 Gray Test - Night), and that look's cube as the
    // person's own LUT.
    let without: Entries = alf4
        .iter()
        .filter(|(path, _)| !path.contains("-1120-"))
        .cloned()
        .collect();
    assert_eq!(without.len(), N - 1);
    let r = import_source(&mut library, &db, &zip_of(work.path(), &(ALF4_ZIP, without)), None);
    assert_eq!(r.added.len(), N - 1, "{r:?}");
    let mine = work.path().join("mine");
    fs::create_dir_all(&mine).unwrap();
    let warm = mine.join("Warm.cube");
    fs::write(&warm, &cubes[2 * 2].1).unwrap();
    import_source(&mut library, &db, &warm, Some(LookInput::LogC4));

    // The cubes: N - 1 look files become cubes, and Warm is the last look.
    let r = import_source(&mut library, &db, &zip_of(work.path(), &logc4_cubes()), None);
    assert_eq!(r.updated.len(), N, "{r:?}");
    assert!(r.added.is_empty(), "{r:?}");
    assert!(
        note_with(&r, "8 looks were replaced by the 3D-LUT version"),
        "the look files only: {:?}",
        r.notes
    );
    assert!(
        note_with(&r, "1 of your own LUTs is a file from ARRI's library"),
        "{:?}",
        r.notes
    );
    assert_eq!(r.renamed.len(), 1, "{:?}", r.renamed);
    check_library(
        &library,
        &Counts {
            arri: N,
            user: 0,
            display: 0,
        },
        "look files and a claim in one import",
    );
}

#[test]
fn look_files_do_not_keep_a_cube_that_is_gone_or_altered_and_the_cubes_bring_it_back() {
    let work = tempfile::tempdir().unwrap();
    let db = small_db(N);
    let mut library = open_library(&work.path().join("app"));
    let c = Counts {
        arri: N,
        user: 0,
        display: 0,
    };
    import_source(&mut library, &db, &zip_of(work.path(), &logc4_cubes()), None);

    // Two of the library's cubes are lost: one deleted, one altered.
    let cubes: Vec<String> = tree(&library.looks_dir())
        .into_iter()
        .filter(|f| f.starts_with("arri/") && f.ends_with(".cube"))
        .collect();
    assert_eq!(cubes.len(), N);
    fs::remove_file(library.looks_dir().join(&cubes[0])).unwrap();
    fs::write(library.looks_dir().join(&cubes[1]), b"not the file any more").unwrap();
    let mut copy = open_library(library.root());
    assert!(!copy.verify(&mut no_progress()).unwrap().is_clean());

    // The look files come: the two looks take them, the others keep their cubes.
    let r = import_source(&mut library, &db, &zip_of(work.path(), &alf4_files()), None);
    assert!(r.added.is_empty(), "{r:?}");
    assert_eq!(r.updated.len(), 2, "{r:?}");
    assert_eq!(r.unchanged.len(), N - 2, "{r:?}");
    assert!(
        note_with(&r, "The 3D LUTs of 2 looks were missing or had changed"),
        "{:?}",
        r.notes
    );
    assert!(
        note_with(&r, "7 looks were already imported from the 3D-LUT version"),
        "{:?}",
        r.notes
    );
    check_library(&library, &c, "look files after two lost cubes");
    let formats = |library: &Library| {
        library
            .looks()
            .iter()
            .filter(|l| l.id.prefix() == "arri")
            .filter(|l| l.lut.as_ref().is_some_and(|u| u.format == LookFormat::Alf4))
            .count()
    };
    assert_eq!(formats(&library), 2, "two looks are the look files now");
    assert!(
        tree(&library.looks_dir())
            .iter()
            .filter(|f| f.ends_with(".cube"))
            .count()
            == N - 2,
        "the altered cube is gone with the lost one"
    );

    // The same look files again change nothing; the cubes bring the two looks back to 3D LUTs.
    let manifest = manifest_bytes(&library);
    let r = import_source(&mut library, &db, &zip_of(work.path(), &alf4_files()), None);
    assert!(r.added.is_empty() && r.updated.is_empty(), "{r:?}");
    assert_eq!(manifest_bytes(&library), manifest);
    let r = import_source(&mut library, &db, &zip_of(work.path(), &logc4_cubes()), None);
    assert_eq!(r.updated.len(), 2, "{r:?}");
    assert!(
        note_with(&r, "2 looks were replaced by the 3D-LUT version"),
        "{:?}",
        r.notes
    );
    check_library(&library, &c, "cubes again");
    assert_eq!(formats(&library), 0, "every look is a 3D LUT again");
}

/// The folder of the LogC4 library for loose-file tests: both libraries' hashes are known, as
/// `tools/looks/fingerprint.py` records ARRI's.
fn known_logc4() -> (PackageDb, Entries, Entries) {
    let (_, cubes) = logc4_cubes();
    let (_, alf4) = alf4_files();
    let db = fingerprinted(
        small_db(N),
        &[
            ("look-library-logc4", 5, &cubes),
            ("look-library-logc4-alf4", ALF_LUT_SIZE, &alf4),
        ],
    );
    (db, cubes, alf4)
}

/// The entry of `entries` whose path ends with `tail`, written as a loose file called `name`.
fn loose(work: &Path, entries: &Entries, tail: &str, name: &str) -> PathBuf {
    let (_, bytes) = entries
        .iter()
        .find(|(path, _)| path.ends_with(tail))
        .unwrap_or_else(|| panic!("{tail}"));
    let dir = work.join("loose");
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join(name);
    fs::write(&path, bytes).unwrap();
    path
}

#[test]
fn a_loose_file_is_named_and_given_an_id_as_it_is_in_the_whole_package() {
    let work = tempfile::tempdir().unwrap();
    let (db, cubes, alf4) = known_logc4();

    // 1111 Gray Test shares its title with 1110: in the package, and so on its own, it is
    // "Gray Test 1111", never plain "Gray Test".
    let mut library = open_library(&work.path().join("app"));
    let one = loose(work.path(), &cubes, "1111 Gray Test.cube", "Whatever I called it.cube");
    let r = import_source(&mut library, &db, &one, None);
    assert_eq!(r.added.len(), 1, "{r:?}");
    assert_eq!(r.added[0].id.to_string(), "arri:1000:gray-test-1111");
    assert_eq!(r.added[0].name, "Gray Test 1111");

    // Two loose files with that title, one after the other: 1110 and 1111, no `-2`.
    let two = loose(work.path(), &cubes, "1110 Gray Test.cube", "Other.cube");
    let r = import_source(&mut library, &db, &two, None);
    assert_eq!(r.added.len(), 1, "{r:?}");
    assert_eq!(r.added[0].id.to_string(), "arri:1000:gray-test-1110");
    assert_eq!(
        ids(&library),
        vec!["arri:1000:gray-test-1110", "arri:1000:gray-test-1111"]
    );
    // A title that is alone in its package stays as it is.
    let alone = loose(work.path(), &cubes, "2110 Studio Sample.cube", "Studio.cube");
    let r = import_source(&mut library, &db, &alone, None);
    assert_eq!(r.added[0].id.to_string(), "arri:2000:studio-sample");

    // The look file of the same look, loose, is that look too: the 3D LUT stays, no second entry.
    let look_file = loose(work.path(), &alf4, "LL-1111-Gray-Test.alf4", "x.alf4");
    let r = import_source(&mut library, &db, &look_file, None);
    assert!(r.added.is_empty() && r.updated.is_empty(), "{r:?}");
    assert_eq!(r.unchanged[0].id.to_string(), "arri:1000:gray-test-1111");
    assert_eq!(ids(&library).len(), 3);
}

#[test]
fn a_loose_cube_after_the_look_files_takes_the_look_over_and_a_loose_look_file_after_the_cubes_does_not() {
    let work = tempfile::tempdir().unwrap();
    let (db, cubes, alf4) = known_logc4();
    let c = |arri| Counts {
        arri,
        user: 0,
        display: 0,
    };

    // The ALF4 zip, then the one cube of a look that shares its title with another.
    let mut library = open_library(&work.path().join("app"));
    import_source(&mut library, &db, &zip_of(work.path(), &alf4_files()), None);
    check_library(&library, &c(N), "the look files");
    let cube = loose(work.path(), &cubes, "1111 Gray Test.cube", "Single.cube");
    let r = import_source(&mut library, &db, &cube, None);
    assert!(r.added.is_empty(), "no second look 1111: {r:?}");
    assert_eq!(r.updated.len(), 1, "{r:?}");
    assert_eq!(r.updated[0].id.to_string(), "arri:1000:gray-test-1111");
    assert!(note_with(&r, "1 look was replaced by the 3D-LUT version"), "{:?}", r.notes);
    check_library(&library, &c(N), "the cube took one look over");

    // The cubes, then one loose look file of such a look.
    let mut library = open_library(&work.path().join("app2"));
    import_source(&mut library, &db, &zip_of(work.path(), &logc4_cubes()), None);
    let look_file = loose(work.path(), &alf4, "LL-1111-Gray-Test.alf4", "Single.alf4");
    let manifest = manifest_bytes(&library);
    let r = import_source(&mut library, &db, &look_file, None);
    assert!(r.added.is_empty() && r.updated.is_empty(), "{r:?}");
    check_library(&library, &c(N), "the look file after the cubes");
    assert_eq!(manifest_bytes(&library), manifest);
}

#[test]
fn a_file_with_the_bytes_of_an_earlier_arri_look_is_still_the_look_its_own_name_says() {
    // ARRI's LogC3 to Rec.709 package holds two files with the same bytes (City and Dream). Each
    // is its own look; a loose copy of one imported first must not swallow the other later.
    let work = tempfile::tempdir().unwrap();
    let (name, mut entries) = rec709_cubes();
    let same = entries
        .iter()
        .find(|(p, _)| p.ends_with("2110-Studio-Sample.cube"))
        .map(|(_, b)| b.clone())
        .unwrap();
    for (path, bytes) in &mut entries {
        if path.ends_with("3110-Sample-A.cube") {
            *bytes = same.clone();
        }
    }
    let db = fingerprinted(small_db(N), &[("look-library-logc3-rec709", 3, &entries)]);
    let mut library = open_library(&work.path().join("app"));

    // The later one of the pair (in the zip's order), alone, first.
    let dream = loose(work.path(), &entries, "3110-Sample-A.cube", "3110-Sample-A.cube");
    let r = import_source(&mut library, &db, &dream, None);
    assert_eq!(r.added.len(), 1, "{r:?}");
    assert_eq!(r.added[0].id.to_string(), "arri:3000:sample-a-logc3-rec709");

    // Then the whole package: the earlier one of the pair is a look of its own, not "already
    // there".
    let package = zip_of(work.path(), &(name, entries));
    let r = import_source(&mut library, &db, &package, None);
    assert_eq!(r.added.len(), N - 1, "{r:?}");
    assert!(
        library
            .get(&LookId::arri(2000, "studio-sample-logc3-rec709"))
            .is_some(),
        "{:?}",
        ids(&library)
    );
    check_library(
        &library,
        &Counts {
            arri: N,
            user: 0,
            display: 0,
        },
        "both files of the pair",
    );
    // And once more: nothing changes.
    let manifest = manifest_bytes(&library);
    let r = import_source(&mut library, &db, &package, None);
    assert!(r.added.is_empty() && r.updated.is_empty(), "{r:?}");
    assert_eq!(manifest_bytes(&library), manifest);
}
