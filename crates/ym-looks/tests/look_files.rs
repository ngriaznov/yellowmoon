//! ARRI Look Files (`.alf4`, `.alf4c`, `.aml`) through recognition, import and the library.
//!
//! Every file here is written by our own writer (`ym_color::alf`) from synthetic LUTs and CDLs; no
//! ARRI file or value is used. Only ARRI-like **names** are, so the recognition rules have
//! something to read. See `support`.

mod support;

use std::fs;
use std::path::Path;

use support::{
    ALF_LUT_SIZE, ALF2_ROOT, ALF2_ZIP, ALF4_ROOT, ALF4_ZIP, Kind, alf4_bytes, alf4_package_files,
    alf4c_bytes, aml_bytes, look_lut, no_progress, open_library, small_db, some_cdl, tree,
    write_alf4_package_zip, write_aml_package_zip, write_package_zip, write_zip,
};
use ym_color::alf::{AlfFile, REC709_COLORSPACE};
use ym_color::cdl::Cdl;
use ym_color::logc::LogC4;
use ym_color::look::{LookInput, LookTransform};
use ym_color::lut::Lut;
use ym_looks::import::{ImportOptions, ImportReport, import, import_path, inspect};
use ym_looks::library::{Library, LookFormat, LookGroup, LookId, LookStatus};
use ym_looks::manifest::{Manifest, sha256_hex};
use ym_looks::recognize::{
    Evidence, Fingerprint, IssueKind, Outcome, PackageDb, PackageStatus, Recognition, Severity,
    Verdict,
};

fn options() -> ImportOptions {
    ImportOptions {
        imported_at: Some("2026-11-02T10:14:00Z".to_string()),
        ..ImportOptions::default()
    }
}

fn recognise(path: &Path, db: &PackageDb) -> Recognition {
    inspect(path, db, None, &mut no_progress()).unwrap()
}

fn import_zip(library: &mut Library, db: &PackageDb, path: &Path) -> (Recognition, ImportReport) {
    import_path(library, db, path, &options(), &mut no_progress()).unwrap()
}

/// The fingerprints of `files`, as `tools/looks/fingerprint.py` records them.
fn fingerprints(files: &[(String, Vec<u8>)]) -> Vec<Fingerprint> {
    files
        .iter()
        .map(|(path, bytes)| Fingerprint {
            path: path.clone(),
            size: bytes.len() as u64,
            lut_size: ALF_LUT_SIZE,
            sha256: sha256_hex(bytes),
        })
        .collect()
}

fn max_diff(a: &LookTransform, b: &LookTransform, steps: usize) -> f64 {
    let mut worst = 0.0_f64;
    for r in 0..steps {
        for g in 0..steps {
            for bl in 0..steps {
                let rgb = [r, g, bl].map(|i| 0.05 + 0.9 * i as f64 / (steps - 1) as f64);
                let (x, y) = (a.apply_own_encoding(rgb), b.apply_own_encoding(rgb));
                for c in 0..3 {
                    worst = worst.max((x[c] - y[c]).abs());
                }
            }
        }
    }
    worst
}

// ---- The ALF4 library ---------------------------------------------------------------------------

#[test]
fn the_alf4_library_is_recognized_as_the_logc4_look_library_by_its_names() {
    let work = tempfile::tempdir().unwrap();
    let n = 9;
    let zip = write_alf4_package_zip(work.path(), n);
    let r = recognise(&zip, &small_db(n));
    assert_eq!(r.verdict(), Verdict::Ready, "{:?}", r.issues);
    assert_eq!(r.packages.len(), 1);
    let finding = &r.packages[0];
    assert_eq!(finding.id, "look-library-logc4-alf4");
    assert_eq!(finding.title, "Look Library · LogC4");
    assert_eq!((finding.found, finding.expected), (n, Some(n)));
    assert_eq!(finding.status, PackageStatus::Complete);
    assert_eq!(finding.evidence, Evidence::Pattern);
    assert_eq!(finding.lut_sizes, vec![ALF_LUT_SIZE]);
    assert_eq!(finding.groups, 3);
    assert_eq!(r.files.len(), n);
    assert!(r.files.iter().all(|f| f.format == LookFormat::Alf4));
    assert_eq!(r.needs_answer_count(), 0);
    assert_eq!(r.rejected_count(), 0);
    // The junk a Mac adds is set aside.
    assert_eq!(r.ignored.macos_metadata, n + 1);

    // Names and groups come from the file names: the LL- prefix and the look code are not part of
    // the name, and the first digit of the code gives the group, named as ARRI names it.
    let mut seen: Vec<(u32, String, String)> = r
        .files
        .iter()
        .map(|f| match &f.outcome {
            Outcome::ArriLook {
                package,
                group_no,
                group_name,
                input,
                name,
                evidence,
                ..
            } => {
                assert_eq!(package, "look-library-logc4-alf4");
                assert_eq!(*input, LookInput::LogC4);
                assert_eq!(*evidence, Evidence::Pattern);
                (*group_no, group_name.clone(), name.clone())
            }
            other => panic!("{}: {other:?}", f.rel_path),
        })
        .collect();
    seen.sort();
    assert_eq!(
        seen[0],
        (1000, "Black and White".to_string(), "Look 01".to_string())
    );
    assert!(seen.contains(&(3000, "Film".to_string(), "Look 02".to_string())));
    assert!(seen.contains(&(4000, "Tinted".to_string(), "Look 03".to_string())));
}

#[test]
fn the_alf4_library_is_recognized_by_hash_whatever_it_is_called() {
    let work = tempfile::tempdir().unwrap();
    let n = 6;
    let files = alf4_package_files(n);
    let mut db = small_db(n);
    db.packages
        .iter_mut()
        .find(|p| p.id == "look-library-logc4-alf4")
        .unwrap()
        .fingerprints = fingerprints(&files);
    // The same files under names that say nothing, in a zip that says nothing.
    let anonymous: Vec<(String, Vec<u8>)> = files
        .iter()
        .enumerate()
        .map(|(i, (_, bytes))| (format!("file{i}.alf4"), bytes.clone()))
        .collect();
    let zip = work.path().join("download.zip");
    write_zip(&zip, &anonymous);
    let r = recognise(&zip, &db);
    assert_eq!(r.verdict(), Verdict::Ready, "{:?}", r.issues);
    assert_eq!(r.packages[0].id, "look-library-logc4-alf4");
    assert_eq!(r.packages[0].evidence, Evidence::Fingerprint);
    assert!(r.files.iter().all(|f| matches!(
        &f.outcome,
        Outcome::ArriLook { evidence: Evidence::Fingerprint, name, .. } if name.starts_with("Look ")
    )));

    // One of them chosen on its own is still that look (and, as with a single `.cube` of a
    // fingerprinted package, the package is said to be incomplete).
    let loose = work.path().join("anything.alf4");
    fs::write(&loose, &files[2].1).unwrap();
    let r = recognise(&loose, &db);
    assert_eq!(r.verdict(), Verdict::Incomplete, "{:?}", r.issues);
    assert!(
        matches!(&r.files[0].outcome, Outcome::ArriLook { name, group_no, .. } if name == "Look 03" && *group_no == 4000),
        "{:?}",
        r.files[0].outcome
    );
    // Without the fingerprint the same file is the person's own look.
    let r = recognise(&loose, &small_db(n));
    assert!(matches!(&r.files[0].outcome, Outcome::OwnLookFile { .. }));
}

#[test]
fn a_folder_or_a_renamed_zip_of_the_alf4_library_is_recognized_by_its_folder_name() {
    let work = tempfile::tempdir().unwrap();
    let n = 6;
    let files = alf4_package_files(n);
    // A zip that says nothing, whose files sit in the library's own folder.
    let zip = work.path().join("download.zip");
    write_zip(&zip, &files);
    let r = recognise(&zip, &small_db(n));
    assert_eq!(
        r.packages[0].id, "look-library-logc4-alf4",
        "{:?}",
        r.issues
    );
    // The unzipped folder, chosen at either level.
    let outer = work.path().join("unzipped");
    for (path, bytes) in &files {
        let target = outer.join(path);
        fs::create_dir_all(target.parent().unwrap()).unwrap();
        fs::write(target, bytes).unwrap();
    }
    for folder in [outer.clone(), outer.join(ALF4_ROOT)] {
        let r = recognise(&folder, &small_db(n));
        assert_eq!(r.packages.len(), 1, "{folder:?}: {:?}", r.issues);
        assert_eq!(r.packages[0].id, "look-library-logc4-alf4");
        assert_eq!(r.packages[0].found, n);
    }
}

#[test]
fn importing_the_alf4_library_keeps_the_files_and_reads_them_as_logc4_looks() {
    let work = tempfile::tempdir().unwrap();
    let n = 9;
    let zip = write_alf4_package_zip(work.path(), n);
    let db = small_db(n);
    let mut library = open_library(&work.path().join("app"));
    let (_, report) = import_zip(&mut library, &db, &zip);
    assert_eq!(report.added.len(), n);
    assert!(report.rejected.is_empty() && report.updated.is_empty() && report.unchanged.is_empty());

    // The ids are the ones the 3D-LUT package gives the same looks.
    let first = LookId::arri(1000, "look-01");
    let look = library.get(&first).expect("look 01");
    assert_eq!(look.name, "Look 01");
    assert_eq!(look.input, LookInput::LogC4);
    assert!(
        matches!(&look.group, LookGroup::Arri { group_no: 1000, group_name } if group_name == "Black and White")
    );
    let lut = look.lut.as_ref().unwrap();
    assert_eq!(lut.format, LookFormat::Alf4);
    assert_eq!(lut.lut_size, ALF_LUT_SIZE);
    assert_eq!(
        look.source.as_ref().and_then(|s| s.package.as_deref()),
        Some("look-library-logc4-alf4")
    );

    // The file in the library is the file in the zip, byte for byte.
    let item = report.added.iter().find(|i| i.id == first).unwrap();
    assert_eq!(item.file, "arri/1000-black-and-white/LL-1010-Look-01.alf4");
    let copy = fs::read(library.looks_dir().join(&item.file)).unwrap();
    assert_eq!(copy, alf4_package_files(n)[0].1);
    assert_eq!(lut.sha256, sha256_hex(&copy));

    // The manifest records the format; reopening reads it back and the look renders.
    let manifest = fs::read_to_string(library.looks_dir().join("manifest.json")).unwrap();
    assert!(manifest.contains("\"format\": \"alf4\""), "{manifest}");
    let reopened = open_library(library.root());
    assert!(reopened.problems().is_empty());
    assert_eq!(
        reopened.get(&first).unwrap().lut.as_ref().unwrap().format,
        LookFormat::Alf4
    );
    let transform = reopened.transform(&first).unwrap();
    assert_eq!(transform.input(), LookInput::LogC4);
    assert!(transform.cdl().is_none(), "the file's CDL is the identity");
    let out = transform
        .apply_logc4(LogC4([0.3, 0.5, 0.7]), 100.0)
        .unwrap();
    let want = look_lut(ALF_LUT_SIZE, 1).sample([0.3, 0.5, 0.7]);
    for c in 0..3 {
        assert!((out.0[c] - want[c]).abs() < 1e-5, "{out:?} {want:?}");
    }
    assert!(library.verify(&mut no_progress()).unwrap().is_clean());

    // Importing it again changes nothing.
    let (_, again) = import_zip(&mut library, &db, &zip);
    assert!(again.added.is_empty() && again.updated.is_empty());
    assert_eq!(again.unchanged.len(), n);
}

#[test]
fn a_changed_or_missing_look_file_is_disabled_like_a_cube() {
    let work = tempfile::tempdir().unwrap();
    let zip = write_alf4_package_zip(work.path(), 3);
    let db = small_db(3);
    let mut library = open_library(&work.path().join("app"));
    import_zip(&mut library, &db, &zip);
    let a = LookId::arri(1000, "look-01");
    let b = LookId::arri(3000, "look-02");
    let dir = library.looks_dir();
    let path_a = dir.join("arri/1000-black-and-white/LL-1010-Look-01.alf4");
    fs::write(&path_a, alf4_bytes(ALF_LUT_SIZE, 99, None)).unwrap();
    fs::remove_file(dir.join("arri/3000-film/LL-3010-Look-02.alf4")).unwrap();
    let report = library.verify(&mut no_progress()).unwrap();
    assert!(!report.is_clean());
    assert_eq!(library.get(&a).unwrap().status, LookStatus::Changed);
    assert_eq!(library.get(&b).unwrap().status, LookStatus::Missing);
    // A changed file is never applied, even before verify has run.
    let fresh = open_library(library.root());
    assert!(fresh.transform(&a).is_err());
    assert!(fresh.transform(&b).is_err());
    assert!(fresh.transform(&LookId::arri(4000, "look-03")).is_ok());
}

#[test]
fn a_look_file_that_was_replaced_by_another_kind_is_refused_when_it_is_built() {
    // The list says `.alf4`; the file is a valid `.aml` with the right checksum (a hand-edited list).
    let work = tempfile::tempdir().unwrap();
    let zip = write_alf4_package_zip(work.path(), 3);
    let db = small_db(3);
    let mut library = open_library(&work.path().join("app"));
    import_zip(&mut library, &db, &zip);
    let id = LookId::arri(1000, "look-01");
    let dir = library.looks_dir();
    let aml = aml_bytes(ALF_LUT_SIZE, 1, None);
    fs::write(
        dir.join("arri/1000-black-and-white/LL-1010-Look-01.alf4"),
        &aml,
    )
    .unwrap();
    let mut manifest = Manifest::load(&dir).unwrap();
    manifest
        .looks
        .iter_mut()
        .find(|l| l.id == id)
        .unwrap()
        .sha256 = sha256_hex(&aml);
    manifest.save(&dir).unwrap();
    let fresh = open_library(library.root());
    let err = fresh.transform(&id).unwrap_err();
    assert!(err.to_string().contains("not usable"), "{err}");
}

// ---- The ALF2 library ---------------------------------------------------------------------------

#[test]
fn the_alf2_library_is_the_logc3_to_rec709_look_library_in_three_strengths() {
    let work = tempfile::tempdir().unwrap();
    let n = 6;
    let zip = write_aml_package_zip(work.path(), n);
    let db = small_db(n);
    let r = recognise(&zip, &db);
    assert_eq!(r.verdict(), Verdict::Ready, "{:?}", r.issues);
    assert_eq!(r.packages.len(), 1);
    let finding = &r.packages[0];
    assert_eq!(finding.id, "look-library-logc3-rec709-alf2");
    assert_eq!(finding.title, "Look Library · LogC3 to Rec.709");
    assert_eq!((finding.found, finding.expected), (3 * n, Some(3 * n)));
    assert_eq!(finding.groups, 3);

    let mut library = open_library(&work.path().join("app"));
    let report = import(&mut library, &db, &zip, &r, &options(), &mut no_progress()).unwrap();
    assert_eq!(report.added.len(), 3 * n);
    for (strength, seed) in [("low", 1), ("medium", 2), ("high", 3)] {
        let id = LookId::arri(1000, &format!("look-01-{strength}-logc3-rec709"));
        let look = library.get(&id).unwrap_or_else(|| panic!("{id}"));
        assert_eq!(
            look.name,
            format!(
                "Look 01 · {}{}",
                &strength[..1].to_uppercase(),
                &strength[1..]
            )
        );
        assert_eq!(look.input, LookInput::LogC3ToRec709);
        assert!(look.includes_display_rendering());
        assert_eq!(look.lut.as_ref().unwrap().format, LookFormat::Aml);
        // Each strength is its own file with its own LUT.
        let transform = library.transform(&id).unwrap();
        let out = transform
            .eval_display_rec709(LogC4([0.4, 0.4, 0.4]))
            .unwrap();
        assert!(out.0.iter().all(|v| v.is_finite()));
        let lut = look_lut(ALF_LUT_SIZE, seed);
        let expected_code = lut.sample(LogC4([0.4, 0.4, 0.4]).to_logc3_ei800().0);
        let expected = expected_code.map(|c| c.powf(2.4));
        for c in 0..3 {
            assert!(
                (out.0[c] - expected[c]).abs() < 1e-4,
                "{strength}: {out:?} {expected:?}"
            );
        }
    }
    let file = report
        .added
        .iter()
        .find(|i| i.id == LookId::arri(1000, "look-01-high-logc3-rec709"))
        .unwrap();
    assert_eq!(
        file.file,
        "arri/1000-black-and-white/LL-1010-Look-01-High-logc3-rec709.aml"
    );
    assert!(library.verify(&mut no_progress()).unwrap().is_clean());
}

#[test]
fn the_misnamed_aml_keeps_its_stem_as_its_name() {
    // ARRI's library holds one file called `...-Beauty-How.aml`: "How" is not a strength.
    let work = tempfile::tempdir().unwrap();
    let zip = work.path().join("looks.zip");
    write_zip(
        &zip,
        &[
            (
                format!("{ALF2_ROOT}/2000 Application/LL-2130-Beauty-How.aml"),
                aml_bytes(ALF_LUT_SIZE, 1, None),
            ),
            (
                format!("{ALF2_ROOT}/2000 Application/LL-2130-Beauty-Medium.aml"),
                aml_bytes(ALF_LUT_SIZE, 2, None),
            ),
            (
                format!("{ALF2_ROOT}/2000 Application/LL-2130-Beauty-High.aml"),
                aml_bytes(ALF_LUT_SIZE, 3, None),
            ),
        ],
    );
    let mut db = small_db(1);
    db.packages
        .iter_mut()
        .find(|p| p.id == "look-library-logc3-rec709-alf2")
        .unwrap()
        .expected
        .cube_files = Some(3);
    let mut library = open_library(&work.path().join("app"));
    let (r, report) = import_zip_path(&mut library, &db, &zip);
    assert_eq!(r.packages[0].id, "look-library-logc3-rec709-alf2");
    let mut names: Vec<&str> = library
        .looks()
        .iter()
        .filter(|l| l.id.prefix() == "arri")
        .map(|l| l.name.as_str())
        .collect();
    names.sort_unstable();
    assert_eq!(names, ["Beauty How", "Beauty · High", "Beauty · Medium"]);
    assert_eq!(report.added.len(), 3);
}

fn import_zip_path(
    library: &mut Library,
    db: &PackageDb,
    path: &Path,
) -> (Recognition, ImportReport) {
    import_path(library, db, path, &options(), &mut no_progress()).unwrap()
}

#[test]
fn looks_with_the_same_title_keep_their_code_in_the_name() {
    let work = tempfile::tempdir().unwrap();
    let zip = work.path().join(ALF2_ZIP);
    let mut files = Vec::new();
    for (code, strength) in [(1110, "Low"), (1111, "Low"), (1110, "High"), (1111, "High")] {
        files.push((
            format!("{ALF2_ROOT}/1000 Black-and-White/LL-{code}-Black-and-White-{strength}.aml"),
            aml_bytes(ALF_LUT_SIZE, code + strength.len() as u32, None),
        ));
    }
    write_zip(&zip, &files);
    let mut db = small_db(1);
    db.packages
        .iter_mut()
        .find(|p| p.id == "look-library-logc3-rec709-alf2")
        .unwrap()
        .expected
        .cube_files = Some(4);
    let r = recognise(&zip, &db);
    let mut names: Vec<String> = r
        .files
        .iter()
        .map(|f| match &f.outcome {
            Outcome::ArriLook { name, .. } => name.clone(),
            other => panic!("{other:?}"),
        })
        .collect();
    names.sort();
    assert_eq!(
        names,
        [
            "Black and White 1110 · High",
            "Black and White 1110 · Low",
            "Black and White 1111 · High",
            "Black and White 1111 · Low"
        ]
    );
}

// ---- The person's own look files ----------------------------------------------------------------

#[test]
fn loose_look_files_import_as_your_own_looks_with_the_input_taken_from_the_file() {
    let work = tempfile::tempdir().unwrap();
    let db = small_db(3);
    let mut library = open_library(&work.path().join("app"));
    let cdl = some_cdl();
    let cases: [(&str, Vec<u8>, LookInput, LookFormat); 3] = [
        (
            "Warm Pull.alf4",
            alf4_bytes(5, 1, Some(&cdl)),
            LookInput::LogC4,
            LookFormat::Alf4,
        ),
        (
            "Night.aml",
            aml_bytes(5, 2, None),
            LookInput::LogC3ToRec709,
            LookFormat::Aml,
        ),
        (
            "Custom.alf4c",
            alf4c_bytes(5, 3, None),
            LookInput::LogC4,
            LookFormat::Alf4c,
        ),
    ];
    for (name, bytes, input, format) in cases {
        let path = work.path().join(name);
        fs::write(&path, &bytes).unwrap();
        let r = recognise(&path, &db);
        // No question, no package: the file says what it is.
        assert_eq!(r.verdict(), Verdict::Ready, "{name}: {:?}", r.issues);
        assert!(r.packages.is_empty() && !r.needs_answer(), "{name}");
        assert_eq!(r.recognized_count(), 1, "{name}");
        assert!(matches!(
            &r.files[0].outcome,
            Outcome::OwnLookFile { input: got, .. } if *got == input
        ));
        assert_eq!(r.files[0].format, format);
        assert_eq!(r.files[0].lut_size, Some(5));
        // `options().input` is not needed and not used.
        let report = import(&mut library, &db, &path, &r, &options(), &mut no_progress()).unwrap();
        assert_eq!(report.added.len(), 1, "{name}");
        let stem = name.rsplit_once('.').unwrap().0;
        let id = report.added[0].id.clone();
        let look = library.get(&id).unwrap();
        assert_eq!(look.name, stem);
        assert_eq!(look.input, input, "{name}");
        assert!(matches!(look.group, LookGroup::User));
        assert_eq!(look.lut.as_ref().unwrap().format, format);
        assert_eq!(report.added[0].file, format!("user/{name}"));
        assert!(library.transform(&id).is_ok(), "{name}");
    }
    assert!(tree(&library.looks_dir()).contains(&"user/Warm Pull.alf4".to_string()));
    assert!(library.verify(&mut no_progress()).unwrap().is_clean());
    assert_eq!(
        library
            .get(&LookId::user("warm-pull"))
            .unwrap()
            .id
            .to_string(),
        "user:warm-pull"
    );
}

#[test]
fn an_imported_look_file_renders_like_the_same_lut_as_a_cube_with_the_same_cdl() {
    let work = tempfile::tempdir().unwrap();
    let db = small_db(3);
    let mut library = open_library(&work.path().join("app"));
    let cdl = some_cdl();
    let path = work.path().join("Graded.alf4");
    fs::write(&path, alf4_bytes(17, 7, Some(&cdl))).unwrap();
    let (_, report) = import_zip_path(&mut library, &db, &path);
    let from_file = library.transform(&report.added[0].id).unwrap();
    let file_cdl: Cdl = *from_file.cdl().expect("the file's CDL");
    // The same LUT as a `.cube`, imported the old way, and the same CDL in front of it.
    let cube = work.path().join("Graded.cube");
    fs::write(&cube, support::cube_text(17, 7, "Graded")).unwrap();
    let with_input = ImportOptions {
        input: Some(LookInput::LogC4),
        ..options()
    };
    let (_, report) =
        import_path(&mut library, &db, &cube, &with_input, &mut no_progress()).unwrap();
    let cube_look = library.transform(&report.added[0].id).unwrap();
    assert!(cube_look.cdl().is_none());
    let reference = LookTransform::new(
        LookInput::LogC4,
        Some(file_cdl),
        cube_look.lut().cloned().map(|l: Lut| l),
    )
    .unwrap();
    let d = max_diff(&from_file, &reference, 9);
    assert!(d < 4e-5, "{d:e}");
    // And the CDL did something: the plain cube is far from the look file.
    assert!(max_diff(&from_file, &cube_look, 9) > 0.01);
    // The file's CDL is what was written (to f32 precision).
    for c in 0..3 {
        assert!((file_cdl.slope[c] - cdl.slope[c]).abs() < 1e-6);
        assert!((file_cdl.power[c] - cdl.power[c]).abs() < 1e-6);
    }
}

#[test]
fn a_folder_or_a_zip_of_your_own_look_files_needs_no_answer() {
    let work = tempfile::tempdir().unwrap();
    let db = small_db(3);
    let files: Vec<(String, Vec<u8>)> = vec![
        ("Mine/A.alf4".to_string(), alf4_bytes(5, 1, None)),
        (
            "Mine/B.alf4".to_string(),
            alf4_bytes(5, 2, Some(&some_cdl())),
        ),
        ("Mine/C.aml".to_string(), aml_bytes(5, 3, None)),
        ("Mine/Guide.pdf".to_string(), b"%PDF".to_vec()),
        ("Mine/old.xml".to_string(), b"<adicam/>".to_vec()),
    ];
    let zip = work.path().join("my looks.zip");
    write_zip(&zip, &files);
    let folder = work.path().join("my looks");
    for (path, bytes) in &files {
        let target = folder.join(path);
        fs::create_dir_all(target.parent().unwrap()).unwrap();
        fs::write(target, bytes).unwrap();
    }
    for source in [&zip, &folder] {
        let r = recognise(source, &db);
        assert_eq!(r.verdict(), Verdict::Ready, "{source:?}: {:?}", r.issues);
        assert!(r.packages.is_empty());
        assert_eq!(r.recognized_count(), 3);
        assert_eq!((r.ignored.pdf, r.ignored.xml), (1, 1));
        let mut library = open_library(&work.path().join(format!(
            "app-{}",
            source.file_name().unwrap().to_string_lossy().replace(' ', "")
        )));
        let report = import(
            &mut library,
            &db,
            source,
            &r,
            &options(),
            &mut no_progress(),
        )
        .unwrap();
        assert_eq!(report.added.len(), 3);
        let mut inputs: Vec<LookInput> = report
            .added
            .iter()
            .map(|i| library.get(&i.id).unwrap().input)
            .collect();
        inputs.sort_by_key(|i| format!("{i:?}"));
        assert_eq!(
            inputs,
            [LookInput::LogC3ToRec709, LookInput::LogC4, LookInput::LogC4]
        );
    }
}

#[test]
fn look_files_beside_cubes_that_need_an_answer_do_not_wait_for_it() {
    let work = tempfile::tempdir().unwrap();
    let db = small_db(3);
    let folder = work.path().join("mixed");
    fs::create_dir_all(&folder).unwrap();
    fs::write(folder.join("Mine.alf4"), alf4_bytes(5, 1, None)).unwrap();
    fs::write(folder.join("Plain.cube"), support::cube_text(5, 2, "Plain")).unwrap();
    let r = recognise(&folder, &db);
    assert_eq!(r.verdict(), Verdict::NeedsAnswer);
    assert_eq!(r.needs_answer_count(), 1, "only the cube asks");
    assert_eq!(r.recognized_count(), 1);
    let mut library = open_library(&work.path().join("app"));
    let with_input = ImportOptions {
        input: Some(LookInput::LogC3Ei800),
        ..options()
    };
    let report = import(
        &mut library,
        &db,
        &folder,
        &r,
        &with_input,
        &mut no_progress(),
    )
    .unwrap();
    assert_eq!(report.added.len(), 2);
    // The answer is for the cube; the look file keeps what it says.
    assert_eq!(
        library.get(&LookId::user("mine")).unwrap().input,
        LookInput::LogC4
    );
    assert_eq!(
        library.get(&LookId::user("plain")).unwrap().input,
        LookInput::LogC3Ei800
    );
}

#[test]
fn an_alf4c_is_read_as_its_cmt_and_says_its_display_transforms_are_not_used() {
    let work = tempfile::tempdir().unwrap();
    let db = small_db(3);
    let path = work.path().join("Custom.alf4c");
    fs::write(&path, alf4c_bytes(9, 4, Some(&some_cdl()))).unwrap();
    let r = recognise(&path, &db);
    assert_eq!(r.verdict(), Verdict::Ready);
    let note = r
        .issues
        .iter()
        .find(|i| i.kind == IssueKind::DisplayTransformsIgnored)
        .expect("a note");
    assert_eq!(note.severity, Severity::Info);
    assert_eq!(
        note.message,
        "Custom.alf4c: its own display transforms are not used; Yellowmoon's display rendering is."
    );
    let mut library = open_library(&work.path().join("app"));
    let report = import(&mut library, &db, &path, &r, &options(), &mut no_progress()).unwrap();
    let transform = library.transform(&report.added[0].id).unwrap();
    assert_eq!(
        transform.lut().unwrap().cube.size(),
        9,
        "the CMT, not a display transform"
    );
    assert!(transform.cdl().is_some());
    assert_eq!(transform.input(), LookInput::LogC4);
}

// ---- Files that cannot be used ------------------------------------------------------------------

fn rejected_reason(r: &Recognition) -> &str {
    match &r.files[0].outcome {
        Outcome::Rejected { reason } => reason,
        other => panic!("{other:?}"),
    }
}

#[test]
fn look_files_that_cannot_be_used_are_refused_in_plain_words() {
    let work = tempfile::tempdir().unwrap();
    let db = small_db(3);

    // An .alf4c with display transforms and no creative look.
    let no_cmt = AlfFile::alf4c(None, None, Some(&look_lut(3, 1)), Some(&look_lut(3, 2)), "")
        .to_bytes()
        .unwrap();
    // An .aml made for a Rec.2020 PQ display.
    let AlfFile::Aml(mut pq) = AlfFile::aml(&look_lut(3, 1), None, "x") else {
        unreachable!()
    };
    pq.colorspace = "com.apple.proapps.colorspace.rec2100.d65.pq".to_owned();
    let pq = AlfFile::Aml(pq).to_bytes().unwrap();
    // An .aml with no LUT.
    let AlfFile::Aml(mut bare) = AlfFile::aml(&look_lut(3, 1), None, "x") else {
        unreachable!()
    };
    bare.size = 0;
    bare.payload.clear();
    let bare = AlfFile::Aml(bare).to_bytes().unwrap();
    // A damaged .alf4 and a cut-short one.
    let good = alf4_bytes(5, 1, None);
    let mut damaged = good.clone();
    let at = damaged.len() - 9;
    damaged[at] ^= 0x40;
    let cut = good[..good.len() - 100].to_vec();
    // An .alf4 whose colour space is not LogC4, and a CDL that is not a number.
    let AlfFile::Alf4(mut other) = AlfFile::alf4(&look_lut(3, 1), None, "", "x") else {
        unreachable!()
    };
    other.lut.colorspace = REC709_COLORSPACE.to_owned();
    let other = AlfFile::Alf4(other).to_bytes().unwrap();
    let AlfFile::Alf4(mut nan) = AlfFile::alf4(&look_lut(3, 1), None, "", "x") else {
        unreachable!()
    };
    nan.params[0].value = f32::NAN;
    let nan = AlfFile::Alf4(nan).to_bytes().unwrap();

    let cases: [(&str, &[u8], &str); 8] = [
        (
            "NoCmt.alf4c",
            &no_cmt,
            "no creative look (CMT), so there is nothing to import",
        ),
        (
            "Wide.aml",
            &pq,
            "It is made for a Rec.2020 PQ display. Yellowmoon can use ARRI Look Files made for Rec.709.",
        ),
        ("Bare.aml", &bare, "It has no LUT"),
        (
            "Damaged.alf4",
            &damaged,
            "Its checksum does not match, so the file is damaged. Download it again.",
        ),
        ("Cut.alf4", &cut, "It is cut short or damaged"),
        ("Other.alf4", &other, "made for another colour space"),
        ("Nan.alf4", &nan, "not a number"),
        (
            "Text.alf4",
            b"LUT_3D_SIZE 2\n",
            "It is not an ARRI Look File",
        ),
    ];
    for (name, bytes, expect) in cases {
        let path = work.path().join(name);
        fs::write(&path, bytes).unwrap();
        let r = recognise(&path, &db);
        assert_eq!(r.verdict(), Verdict::Blocked, "{name}");
        let reason = rejected_reason(&r);
        assert!(reason.contains(expect), "{name}: {reason}");
        let issue = &r.issues[0];
        assert_eq!(issue.kind, IssueKind::AllRejected, "{name}");
        assert!(
            issue
                .message
                .starts_with(&format!("{name} cannot be used. ")),
            "{name}: {}",
            issue.message
        );
        // Importing it is refused with the same words, and writes nothing.
        let mut library = open_library(&work.path().join(format!("app-{name}")));
        let err = import(&mut library, &db, &path, &r, &options(), &mut no_progress()).unwrap_err();
        assert!(err.to_string().contains(expect), "{name}: {err}");
        assert!(tree(&library.looks_dir()).is_empty(), "{name}");
    }
}

#[test]
fn one_bad_look_file_does_not_stop_the_good_ones() {
    let work = tempfile::tempdir().unwrap();
    let db = small_db(3);
    let folder = work.path().join("looks");
    fs::create_dir_all(&folder).unwrap();
    fs::write(folder.join("Good.alf4"), alf4_bytes(5, 1, None)).unwrap();
    let mut bad = alf4_bytes(5, 2, None);
    let at = bad.len() - 5;
    bad[at] ^= 0x01;
    fs::write(folder.join("Bad.alf4"), bad).unwrap();
    let mut library = open_library(&work.path().join("app"));
    let (r, report) = import_zip_path(&mut library, &db, &folder);
    assert_eq!(r.verdict(), Verdict::Ready);
    assert_eq!(report.added.len(), 1);
    assert_eq!(report.rejected.len(), 1);
    assert_eq!(report.rejected[0].name, "Bad.alf4");
    assert!(
        report.rejected[0].reason.contains("damaged")
            || report.rejected[0].reason.contains("cut short")
    );
    assert!(r.issues.iter().any(
        |i| i.kind == IssueKind::Rejected && i.message.starts_with("Bad.alf4 cannot be used.")
    ));
}

#[test]
fn arris_older_xml_look_files_are_named_and_not_read() {
    let work = tempfile::tempdir().unwrap();
    let db = small_db(3);
    let xml = work.path().join("Classic.xml");
    fs::write(&xml, b"<?xml version=\"1.0\"?><adicam version=\"1.0\"/>").unwrap();
    let r = recognise(&xml, &db);
    assert_eq!(r.verdict(), Verdict::Blocked);
    assert_eq!(r.issues[0].kind, IssueKind::NoLookFiles);
    assert!(
        r.issues[0].message.contains("ALF-1"),
        "{}",
        r.issues[0].message
    );
    let zip = work.path().join("classic.zip");
    write_zip(&zip, &[("Look/a.xml".to_string(), b"<adicam/>".to_vec())]);
    let r = recognise(&zip, &db);
    assert_eq!(r.ignored.xml, 1);
    assert!(r.issues[0].message.contains("older look files"));
}

// ---- One look, one entry ------------------------------------------------------------------------

#[test]
fn a_cube_version_of_the_logc4_library_is_kept_when_the_look_files_come_later() {
    let work = tempfile::tempdir().unwrap();
    let n = 6;
    let db = small_db(n);
    let mut library = open_library(&work.path().join("app"));
    let cubes = write_package_zip(work.path(), Kind::LookLogC4, n);
    let (_, first) = import_zip(&mut library, &db, &cubes);
    assert_eq!(first.added.len(), n);
    let ids_before: Vec<String> = library.looks().iter().map(|l| l.id.to_string()).collect();

    let alf = write_alf4_package_zip(work.path(), n);
    let (r, second) = import_zip(&mut library, &db, &alf);
    assert_eq!(r.packages[0].id, "look-library-logc4-alf4");
    assert!(
        second.added.is_empty() && second.updated.is_empty(),
        "{second:?}"
    );
    assert_eq!(second.unchanged.len(), n);
    assert!(
        second
            .notes
            .iter()
            .any(|m| m.contains("already imported from the 3D-LUT version")),
        "{:?}",
        second.notes
    );
    // Nothing changed in the library: the same ids, all cubes, no look file on disk.
    let ids_after: Vec<String> = library.looks().iter().map(|l| l.id.to_string()).collect();
    assert_eq!(ids_before, ids_after);
    assert!(
        library
            .looks()
            .iter()
            .filter_map(|l| l.lut.as_ref())
            .all(|l| l.format.is_cube())
    );
    assert!(
        tree(&library.looks_dir())
            .iter()
            .all(|p| !p.ends_with(".alf4"))
    );
}

#[test]
fn a_cube_version_replaces_the_look_files_of_the_same_looks() {
    let work = tempfile::tempdir().unwrap();
    let n = 6;
    let db = small_db(n);
    let mut library = open_library(&work.path().join("app"));
    let alf = write_alf4_package_zip(work.path(), n);
    let (_, first) = import_zip(&mut library, &db, &alf);
    assert_eq!(first.added.len(), n);
    assert!(
        tree(&library.looks_dir())
            .iter()
            .any(|p| p.ends_with(".alf4"))
    );

    let cubes = write_package_zip(work.path(), Kind::LookLogC4, n);
    let (_, second) = import_zip(&mut library, &db, &cubes);
    assert!(second.added.is_empty(), "{second:?}");
    assert_eq!(second.updated.len(), n);
    assert!(
        second
            .notes
            .iter()
            .any(|m| m.contains("replaced by the 3D-LUT version")),
        "{:?}",
        second.notes
    );
    // One entry per look, now `.cube`; the look files are gone from the folder.
    assert_eq!(
        library
            .looks()
            .iter()
            .filter(|l| l.id.prefix() == "arri")
            .count(),
        n
    );
    assert!(
        library
            .looks()
            .iter()
            .filter_map(|l| l.lut.as_ref())
            .all(|l| l.format.is_cube())
    );
    let files = tree(&library.looks_dir());
    assert!(files.iter().all(|p| !p.ends_with(".alf4")), "{files:?}");
    assert!(library.verify(&mut no_progress()).unwrap().is_clean());
    let manifest = fs::read_to_string(library.looks_dir().join("manifest.json")).unwrap();
    assert!(
        !manifest.contains("\"format\""),
        "cubes carry no format key: {manifest}"
    );
    // The renewed library opens and renders.
    let reopened = open_library(library.root());
    assert!(reopened.problems().is_empty());
    for look in reopened.looks().iter().filter(|l| l.id.prefix() == "arri") {
        assert!(reopened.transform(&look.id).is_ok(), "{}", look.id);
    }
}

#[test]
fn the_two_logc3_libraries_and_the_logc4_look_files_never_collide() {
    let work = tempfile::tempdir().unwrap();
    let n = 3;
    let db = small_db(n);
    let mut library = open_library(&work.path().join("app"));
    let mut total = 0;
    for zip in [
        write_package_zip(work.path(), Kind::LookLogC3Rec709, n),
        write_aml_package_zip(work.path(), n),
        write_alf4_package_zip(work.path(), n),
        write_package_zip(work.path(), Kind::LookLogC3, n),
    ] {
        let (_, report) = import_zip(&mut library, &db, &zip);
        assert!(
            report.updated.is_empty() && report.unchanged.is_empty(),
            "{zip:?}"
        );
        total += report.added.len();
    }
    // 3 cubes (Rec.709) + 9 strengths + 3 look files + 3 cubes (LogC3).
    assert_eq!(total, 3 + 9 + 3 + 3);
    assert_eq!(
        library
            .looks()
            .iter()
            .filter(|l| l.id.prefix() == "arri")
            .count(),
        total
    );
    assert!(library.verify(&mut no_progress()).unwrap().is_clean());
}

#[test]
fn the_same_own_look_file_again_changes_nothing_and_another_one_gets_a_number() {
    let work = tempfile::tempdir().unwrap();
    let db = small_db(3);
    let mut library = open_library(&work.path().join("app"));
    let path = work.path().join("Mine.alf4");
    fs::write(&path, alf4_bytes(5, 1, None)).unwrap();
    let (_, first) = import_zip_path(&mut library, &db, &path);
    assert_eq!(first.added.len(), 1);
    let (_, again) = import_zip_path(&mut library, &db, &path);
    assert!(again.added.is_empty() && again.updated.is_empty());
    assert_eq!(again.unchanged.len(), 1);
    // Different content under the same name never overwrites the first.
    fs::write(&path, alf4_bytes(5, 2, None)).unwrap();
    let (_, other) = import_zip_path(&mut library, &db, &path);
    assert_eq!(other.added[0].id.to_string(), "user:mine-2");
    assert_eq!(other.added[0].file, "user/Mine-2.alf4");
    // And a `.cube` with the same name sits beside them.
    let cube = work.path().join("Mine.cube");
    fs::write(&cube, support::cube_text(5, 3, "Mine")).unwrap();
    let with_input = ImportOptions {
        input: Some(LookInput::LogC4),
        ..options()
    };
    let (_, c) = import_path(&mut library, &db, &cube, &with_input, &mut no_progress()).unwrap();
    assert_eq!(c.added[0].id.to_string(), "user:mine-3");
    assert!(library.verify(&mut no_progress()).unwrap().is_clean());
}

#[test]
fn a_look_file_renamed_to_the_wrong_extension_is_stored_as_what_it_is() {
    let work = tempfile::tempdir().unwrap();
    let db = small_db(3);
    let mut library = open_library(&work.path().join("app"));
    // An .aml's bytes under an .alf4 name.
    let path = work.path().join("Night.alf4");
    fs::write(&path, aml_bytes(5, 2, None)).unwrap();
    let r = recognise(&path, &db);
    assert_eq!(r.files[0].format, LookFormat::Aml);
    let report = import(&mut library, &db, &path, &r, &options(), &mut no_progress()).unwrap();
    assert_eq!(report.added[0].file, "user/Night.aml");
    let look = library.get(&report.added[0].id).unwrap();
    assert_eq!(look.input, LookInput::LogC3ToRec709);
    assert!(library.transform(&look.id).is_ok());
}

#[test]
fn old_lists_without_a_format_still_read_as_cubes() {
    let work = tempfile::tempdir().unwrap();
    let db = small_db(3);
    let mut library = open_library(&work.path().join("app"));
    let cube = work.path().join("Old.cube");
    fs::write(&cube, support::cube_text(5, 1, "Old")).unwrap();
    let with_input = ImportOptions {
        input: Some(LookInput::LogC4),
        ..options()
    };
    import_path(&mut library, &db, &cube, &with_input, &mut no_progress()).unwrap();
    let text = fs::read_to_string(library.looks_dir().join("manifest.json")).unwrap();
    assert!(
        !text.contains("format"),
        "a list of cubes is what 0.1.1 wrote: {text}"
    );
    let reopened = open_library(library.root());
    let look = reopened.get(&LookId::user("old")).unwrap();
    assert_eq!(look.lut.as_ref().unwrap().format, LookFormat::Cube);
    assert!(reopened.transform(&look.id).is_ok());
}

#[test]
fn packages_of_look_files_have_what_the_recogniser_reads() {
    // The shipped list (with its fingerprints): the look-file packages are looks, of their own kind.
    let db = PackageDb::load(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/looks/arri-packages.json"),
    )
    .unwrap();
    let alf4 = db.package("look-library-logc4-alf4").unwrap();
    assert_eq!(alf4.format, LookFormat::Alf4);
    assert_eq!(alf4.fingerprints.len(), 87);
    assert_eq!(alf4.expected.lut_sizes, [33]);
    assert_eq!(
        alf4.download_title(),
        "Look Library for LogC4 (ALF4 Look Files)"
    );
    assert_eq!(alf4.look_input(), Some(LookInput::LogC4));
    let aml = db.package("look-library-logc3-rec709-alf2").unwrap();
    assert_eq!(aml.format, LookFormat::Aml);
    assert_eq!(aml.fingerprints.len(), 261);
    assert_eq!(aml.download_title(), "ARRI Look Library LogC3 (Look Files)");
    assert_eq!(aml.look_input(), Some(LookInput::LogC3ToRec709));
    assert!(
        aml.fingerprints
            .iter()
            .all(|f| f.path.ends_with(".aml") && f.lut_size == 33)
    );
    assert!(
        alf4.fingerprints
            .iter()
            .all(|f| f.path.ends_with(".alf4") && f.lut_size == 33)
    );
    // The 3D-LUT packages are untouched.
    assert_eq!(
        db.package("look-library-logc4").unwrap().format,
        LookFormat::Cube
    );
    assert_eq!(
        db.package("look-library-logc4").unwrap().fingerprints.len(),
        87
    );
    // The Downloads watcher knows both look-file zips by the names ARRI gives them, and the 3D-LUT
    // zips are still told apart from them.
    let for_zip = |name: &str| -> Vec<String> {
        db.packages_for_zip_name(name)
            .iter()
            .map(|p| p.id.clone())
            .collect()
    };
    assert_eq!(for_zip(ALF4_ZIP), ["look-library-logc4-alf4"]);
    assert_eq!(for_zip(ALF2_ZIP), ["look-library-logc3-rec709-alf2"]);
    assert_eq!(
        for_zip("ARRI Look Library LogC4 (log-to-log) 3D-LUTs_data.zip"),
        ["look-library-logc4"]
    );
    assert_eq!(
        for_zip("ARRI Look Library LogC3 (to Rec709) 3D-LUTs_data.zip"),
        ["look-library-logc3-rec709"]
    );
    assert_eq!(
        for_zip("ARRI Look Library LogC3 (log-to-log) 3D LUTs_data.zip"),
        ["look-library-logc3"]
    );
}
