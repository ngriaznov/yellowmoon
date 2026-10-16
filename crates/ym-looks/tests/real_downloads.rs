//! The recogniser and the importer against ARRI's real downloads. **Ignored by default**: the files
//! are ARRI's (all rights reserved), so they never live in the repository and CI never needs them.
//!
//! ```text
//! YM_ARRI_DOWNLOADS=/path/to/folder cargo test -p ym-looks --test real_downloads --release \
//!     -- --ignored --nocapture
//! ```
//!
//! The folder holds the five public downloads under these names: `logc4.zip` (Look Library for
//! LogC4, 3D-LUTs), `logc3.zip` (LogC3 log-to-log, 3D LUTs), `logc3-rec709.zip` (LogC3 to Rec709,
//! 3D-LUTs), `alf4.zip` (the LogC4 look files for the camera) and `alf2.zip` (the LogC3 ones).
//! The test works on copies under a temporary folder, with the file names ARRI's site gives the
//! downloads, and prints only names, counts and differences.
//!
//! Gates for the look-file libraries (`docs/08-looks.md` section ARRI Look Files): every `.alf4`
//! LUT equals the even nodes of the matching 65-point cube within 7.63e-6 (87 of 87), all 261
//! `.aml` files parse and import, and every file is rebuilt byte for byte from what was read.

use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};

use ym_color::logc::LogC4;
use ym_color::look::LookInput;
use ym_looks::import::{ImportOptions, import, import_path, inspect};
use ym_looks::library::Library;
use ym_looks::recognize::{Evidence, Outcome, PackageDb, PackageStatus, Recognition, Verdict};

mod support;
use support::{no_progress, open_library};

/// (file in the downloads folder, the name ARRI's site gives it, package id).
const LUT_ZIPS: [(&str, &str, &str); 3] = [
    (
        "logc4.zip",
        "ARRI Look Library LogC4 (log-to-log) 3D-LUTs_data.zip",
        "look-library-logc4",
    ),
    (
        "logc3.zip",
        "ARRI Look Library LogC3 (log-to-log) 3D LUTs_data.zip",
        "look-library-logc3",
    ),
    (
        "logc3-rec709.zip",
        "ARRI Look Library LogC3 (to Rec709) 3D-LUTs_data.zip",
        "look-library-logc3-rec709",
    ),
];

/// The look-file packages: file, the name ARRI's site gives it, package id, files, extension.
const LOOK_FILE_ZIPS: [(&str, &str, &str, usize, &str); 2] = [
    (
        "alf4.zip",
        "arri-look-library-logc4-log-to-log-alf4-data.zip",
        "look-library-logc4-alf4",
        87,
        "alf4",
    ),
    (
        "alf2.zip",
        "ARRI Look Library LogC3 (log-to-Rec709) ALF2_data.zip",
        "look-library-logc3-rec709-alf2",
        261,
        "aml",
    ),
];

fn downloads() -> Option<PathBuf> {
    let Some(dir) = std::env::var_os("YM_ARRI_DOWNLOADS") else {
        eprintln!("YM_ARRI_DOWNLOADS is not set: nothing to check");
        return None;
    };
    Some(PathBuf::from(dir))
}

fn shipped_db() -> PackageDb {
    PackageDb::load(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/looks/arri-packages.json"),
    )
    .expect("arri-packages.json")
}

/// The shipped list without its fingerprints: names and headers only, as before M3-H2.
fn patterns_only() -> PackageDb {
    let mut db = shipped_db();
    for package in &mut db.packages {
        package.fingerprints.clear();
    }
    db
}

/// A copy of `zip` under `dir` with the name ARRI's site gives it (a symbolic link: the file is
/// only read).
fn linked(dir: &Path, zip: &Path, name: &str) -> PathBuf {
    let path = dir.join(name);
    #[cfg(unix)]
    std::os::unix::fs::symlink(zip, &path).expect("symlink");
    #[cfg(not(unix))]
    fs::copy(zip, &path).expect("copy");
    path
}

/// Unzips into `dir/<name without .zip>`, as Finder does, and returns that folder.
fn unzip_like_finder(zip: &Path, dir: &Path, name: &str) -> PathBuf {
    let folder = dir.join(name.trim_end_matches(".zip"));
    let mut archive = zip::ZipArchive::new(File::open(zip).expect("open")).expect("zip");
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i).expect("entry");
        let Some(rel) = entry.enclosed_name() else {
            continue;
        };
        let target = folder.join(rel);
        if entry.is_dir() {
            fs::create_dir_all(&target).expect("mkdir");
        } else {
            fs::create_dir_all(target.parent().expect("parent")).expect("mkdir");
            io::copy(&mut entry, &mut File::create(&target).expect("create")).expect("copy");
        }
    }
    folder
}

/// The one folder inside a Finder-unzipped folder (ARRI's zips hold a single top folder).
fn inner_folder(folder: &Path) -> PathBuf {
    let dirs: Vec<PathBuf> = fs::read_dir(folder)
        .expect("read")
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir() && p.file_name().is_some_and(|n| n != "__MACOSX"))
        .collect();
    assert_eq!(dirs.len(), 1, "{dirs:?}");
    dirs[0].clone()
}

fn check_look_package(label: &str, r: &Recognition, package: &str, by_hash: bool) {
    assert_eq!(r.verdict(), Verdict::Ready, "{label}: {:?}", r.issues);
    assert_eq!(r.packages.len(), 1, "{label}: {:?}", r.packages);
    let finding = &r.packages[0];
    assert_eq!(finding.id, package, "{label}");
    assert_eq!((finding.found, finding.expected), (87, Some(87)), "{label}");
    assert_eq!(finding.status, PackageStatus::Complete, "{label}");
    assert_eq!(
        finding.groups, 9,
        "{label}: nine groups from the first digit of the look code"
    );
    assert_eq!(r.files.len(), 87, "{label}");
    assert_eq!(r.needs_answer_count(), 0, "{label}");
    assert_eq!(r.rejected_count(), 0, "{label}");
    assert!(
        r.files.iter().all(
            |f| matches!(&f.outcome, Outcome::ArriLook { package: p, evidence, .. }
            if p == package && (*evidence == Evidence::Fingerprint) == by_hash)
        ),
        "{label}"
    );
    assert_eq!(
        finding.evidence,
        if by_hash {
            Evidence::Fingerprint
        } else {
            Evidence::Pattern
        },
        "{label}"
    );
    // Groups and names come out as ARRI's.
    let mut groups: Vec<(u32, String)> = r
        .files
        .iter()
        .filter_map(|f| match &f.outcome {
            Outcome::ArriLook {
                group_no,
                group_name,
                ..
            } => Some((*group_no, group_name.clone())),
            _ => None,
        })
        .collect();
    groups.sort();
    groups.dedup();
    let names: Vec<&str> = groups.iter().map(|g| g.1.as_str()).collect();
    assert_eq!(
        names,
        [
            "Black and White",
            "Application",
            "Film",
            "Tinted",
            "Environment",
            "Season",
            "Contrast",
            "Period",
            "Special"
        ],
        "{label}"
    );
}

#[test]
#[ignore = "needs ARRI's real downloads: set YM_ARRI_DOWNLOADS (see the file header)"]
fn the_three_3d_lut_packages_are_recognized_as_downloaded_and_unzipped() {
    let Some(source) = downloads() else {
        return;
    };
    for (mode, db) in [("by hash", shipped_db()), ("by names", patterns_only())] {
        let by_hash = mode == "by hash";
        if by_hash {
            assert!(
                db.packages
                    .iter()
                    .filter(|p| p.role == ym_looks::recognize::PackageRole::Looks)
                    .all(|p| p.is_fingerprinted()),
                "the shipped list has no fingerprints for a looks package"
            );
        }
        for (file, name, package) in LUT_ZIPS {
            let work = tempfile::tempdir().unwrap();
            let zip = source.join(file);
            let start = std::time::Instant::now();

            // 1. The zip as downloaded.
            let r = inspect(
                &linked(work.path(), &zip, name),
                &db,
                None,
                &mut no_progress(),
            )
            .unwrap();
            check_look_package(&format!("{mode}: {name}"), &r, package, by_hash);
            eprintln!("{mode}: {name}: {package}, 87 files, {:?}", start.elapsed());
            // The junk of a Mac zip is set aside, never read as a LUT.
            assert!(
                r.ignored.macos_metadata == 0 || r.ignored.macos_metadata >= 87,
                "{:?}",
                r.ignored
            );

            // 2. The folder Finder makes, and the folder inside it.
            let outer = unzip_like_finder(&zip, work.path(), name);
            let inner = inner_folder(&outer);
            for folder in [&outer, &inner] {
                let label = format!(
                    "{mode}: folder {}",
                    folder.file_name().unwrap().to_string_lossy()
                );
                let r = inspect(folder, &db, None, &mut no_progress()).unwrap();
                check_look_package(&label, &r, package, by_hash);
            }

            // 3. The zip under a name that says nothing: the folder inside it still does (names)
            // and the hashes always do.
            let r = inspect(
                &linked(work.path(), &zip, "download.zip"),
                &db,
                None,
                &mut no_progress(),
            )
            .unwrap();
            check_look_package(&format!("{mode}: download.zip"), &r, package, by_hash);
        }
    }
}

/// The look-file libraries are the looks of their package, in every layout, by hash and by names.
fn check_look_file_package(
    label: &str,
    r: &Recognition,
    package: &str,
    files: usize,
    by_hash: bool,
) {
    assert_eq!(r.verdict(), Verdict::Ready, "{label}: {:?}", r.issues);
    assert_eq!(r.packages.len(), 1, "{label}: {:?}", r.packages);
    let finding = &r.packages[0];
    assert_eq!(finding.id, package, "{label}");
    assert_eq!(
        (finding.found, finding.expected),
        (files, Some(files)),
        "{label}"
    );
    assert_eq!(finding.status, PackageStatus::Complete, "{label}");
    assert_eq!(
        finding.groups, 9,
        "{label}: nine groups from the first digit of the look code"
    );
    assert_eq!(finding.lut_sizes, vec![33], "{label}");
    assert_eq!(r.files.len(), files, "{label}");
    assert_eq!(r.needs_answer_count(), 0, "{label}");
    assert_eq!(r.rejected_count(), 0, "{label}");
    assert!(
        r.files.iter().all(
            |f| matches!(&f.outcome, Outcome::ArriLook { package: p, evidence, .. }
            if p == package && (*evidence == Evidence::Fingerprint) == by_hash)
        ),
        "{label}"
    );
    let mut names: Vec<(u32, String)> = r
        .files
        .iter()
        .filter_map(|f| match &f.outcome {
            Outcome::ArriLook {
                group_no,
                group_name,
                ..
            } => Some((*group_no, group_name.clone())),
            _ => None,
        })
        .collect();
    names.sort();
    names.dedup();
    let groups: Vec<&str> = names.iter().map(|g| g.1.as_str()).collect();
    assert_eq!(
        groups,
        [
            "Black and White",
            "Application",
            "Film",
            "Tinted",
            "Environment",
            "Season",
            "Contrast",
            "Period",
            "Special"
        ],
        "{label}"
    );
}

#[test]
#[ignore = "needs ARRI's real downloads: set YM_ARRI_DOWNLOADS (see the file header)"]
fn the_look_file_libraries_are_recognized_as_downloaded_and_unzipped() {
    let Some(source) = downloads() else {
        return;
    };
    for (mode, db) in [("by hash", shipped_db()), ("by names", patterns_only())] {
        let by_hash = mode == "by hash";
        for (file, name, package, files, extension) in LOOK_FILE_ZIPS {
            let work = tempfile::tempdir().unwrap();
            let zip = source.join(file);
            let start = std::time::Instant::now();

            // 1. The zip as downloaded.
            let r = inspect(
                &linked(work.path(), &zip, name),
                &db,
                None,
                &mut no_progress(),
            )
            .unwrap();
            check_look_file_package(&format!("{mode}: {name}"), &r, package, files, by_hash);
            eprintln!(
                "{mode}: {name}: {package}, {files} files, {:?}",
                start.elapsed()
            );

            // 2. The folder Finder makes, and the folder inside it; and the zip under a name that
            // says nothing (the folder inside it, or the hashes, still do).
            let outer = unzip_like_finder(&zip, work.path(), name);
            let inner = inner_folder(&outer);
            for folder in [&outer, &inner] {
                let label = format!(
                    "{mode}: folder {}",
                    folder.file_name().unwrap().to_string_lossy()
                );
                let r = inspect(folder, &db, None, &mut no_progress()).unwrap();
                check_look_file_package(&label, &r, package, files, by_hash);
            }
            let r = inspect(
                &linked(work.path(), &zip, "download.zip"),
                &db,
                None,
                &mut no_progress(),
            )
            .unwrap();
            check_look_file_package(
                &format!("{mode}: download.zip"),
                &r,
                package,
                files,
                by_hash,
            );

            // 3. One file chosen by itself: that look by hash, the person's own look by names.
            let loose = walk_first(&outer, &[extension]).expect("a look file");
            let r = inspect(&loose, &db, None, &mut no_progress()).unwrap();
            assert_eq!(r.files.len(), 1);
            match (&r.files[0].outcome, by_hash) {
                (Outcome::ArriLook { package: p, .. }, true) => assert_eq!(p, package),
                (Outcome::OwnLookFile { input, .. }, false) => {
                    assert_eq!(
                        *input,
                        if extension == "aml" {
                            LookInput::LogC3ToRec709
                        } else {
                            LookInput::LogC4
                        }
                    );
                }
                (other, _) => panic!("{mode}: loose {extension}: {other:?}"),
            }
        }
    }
}

/// Reads every `*.ext` file of a zip as (path, bytes), skipping the Mac junk.
fn zip_files(zip: &Path, extension: &str) -> Vec<(String, Vec<u8>)> {
    use std::io::Read;
    let mut archive = zip::ZipArchive::new(File::open(zip).expect("open")).expect("zip");
    let mut out = Vec::new();
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i).expect("entry");
        let Some(name) = entry
            .enclosed_name()
            .map(|p| p.to_string_lossy().replace('\\', "/"))
        else {
            continue;
        };
        if entry.is_dir()
            || name.contains("__MACOSX")
            || name.rsplit('/').next().is_some_and(|n| n.starts_with("._"))
            || !name
                .to_ascii_lowercase()
                .ends_with(&format!(".{extension}"))
        {
            continue;
        }
        let mut bytes = Vec::new();
        entry.read_to_end(&mut bytes).expect("read");
        out.push((name, bytes));
    }
    out.sort();
    out
}

fn look_code_of(path: &str) -> u32 {
    use ym_looks::recognize::names::{file_name, look_code};
    let name = file_name(path);
    look_code(name.rsplit_once('.').map_or(name, |(stem, _)| stem))
        .unwrap_or_else(|| panic!("no look code in {path}"))
}

#[test]
#[ignore = "needs ARRI's real downloads: set YM_ARRI_DOWNLOADS (see the file header)"]
fn every_alf4_lut_is_the_even_nodes_of_the_matching_65_point_cube() {
    use ym_color::alf::{self, AlfFile};
    use ym_color::cube;
    let Some(source) = downloads() else {
        return;
    };
    let alf4 = zip_files(&source.join("alf4.zip"), "alf4");
    let cubes = zip_files(&source.join("logc4.zip"), "cube");
    assert_eq!((alf4.len(), cubes.len()), (87, 87));
    let cube_of = |code: u32| {
        let matching: Vec<_> = cubes
            .iter()
            .filter(|(p, _)| look_code_of(p) == code)
            .collect();
        assert_eq!(matching.len(), 1, "look code {code}");
        cube::parse_bytes(&matching[0].1)
            .expect("cube parses")
            .lut3d
            .expect("3D cube")
    };
    let (mut matched, mut worst, mut rebuilt) = (0, 0.0_f64, 0);
    for (path, bytes) in &alf4 {
        let code = look_code_of(path);
        let look = alf::parse(bytes).unwrap_or_else(|e| panic!("{path}: {e}"));
        assert_eq!(look.input, LookInput::LogC4, "{path}");
        assert_eq!(look.lut.size(), 33, "{path}");
        assert!(
            look.cdl.is_none(),
            "{path}: library files carry the identity CDL"
        );
        let cube = cube_of(code);
        assert_eq!(cube.size(), 65, "{path}");
        let mut file_worst = 0.0_f64;
        for r in 0..33 {
            for g in 0..33 {
                for b in 0..33 {
                    let a = look.lut.entry(r, g, b).unwrap();
                    let c = cube.entry(2 * r, 2 * g, 2 * b).unwrap();
                    for k in 0..3 {
                        file_worst = file_worst.max((a[k] - c[k]).abs());
                    }
                }
            }
        }
        assert!(file_worst <= 7.63e-6, "{path}: {file_worst:e}");
        worst = worst.max(file_worst);
        matched += 1;
        // The writer rebuilds the file from what was read.
        let file = alf::parse_file(bytes).unwrap();
        assert!(matches!(file, AlfFile::Alf4(_)));
        if file.to_bytes().unwrap() == *bytes {
            rebuilt += 1;
        }
    }
    eprintln!(
        "{matched}/87 .alf4 LUTs equal the even nodes of the 65-point cube, worst {worst:.4e} (bound 7.63e-6); {rebuilt}/87 rebuilt byte for byte"
    );
    assert_eq!(matched, 87);
    assert_eq!(rebuilt, 87);
}

#[test]
#[ignore = "needs ARRI's real downloads: set YM_ARRI_DOWNLOADS (see the file header)"]
fn all_261_aml_files_parse_rebuild_and_follow_the_rec709_cubes() {
    use ym_color::alf::{self, AlfFile};
    use ym_color::cube;
    let Some(source) = downloads() else {
        return;
    };
    let amls = zip_files(&source.join("alf2.zip"), "aml");
    assert_eq!(amls.len(), 261);
    let cubes = zip_files(&source.join("logc3-rec709.zip"), "cube");
    let (mut parsed, mut rebuilt) = (0, 0);
    let mut high_means: Vec<f64> = Vec::new();
    for (path, bytes) in &amls {
        let look = alf::parse(bytes).unwrap_or_else(|e| panic!("{path}: {e}"));
        assert_eq!(look.input, LookInput::LogC3ToRec709, "{path}");
        assert_eq!(look.lut.size(), 33, "{path}");
        assert!(
            look.cdl.is_none(),
            "{path}: library files carry the identity CDL"
        );
        parsed += 1;
        let file = alf::parse_file(bytes).unwrap();
        assert!(matches!(file, AlfFile::Aml(_)));
        if file.to_bytes().unwrap() == *bytes {
            rebuilt += 1;
        }
        // The High strength is the closest to the 3D-LUT version of the same look.
        if path.ends_with("-High.aml") {
            let code = look_code_of(path);
            let matching: Vec<_> = cubes
                .iter()
                .filter(|(p, _)| look_code_of(p) == code)
                .collect();
            if let [(_, cube_bytes)] = matching.as_slice() {
                let cube = cube::parse_bytes(cube_bytes).unwrap().lut3d.unwrap();
                assert_eq!(cube.size(), 33);
                let mut total = 0.0;
                for (a, c) in look.lut.data().iter().zip(cube.data()) {
                    for k in 0..3 {
                        total += (a[k] - c[k]).abs();
                    }
                }
                high_means.push(total / (3.0 * look.lut.data().len() as f64));
            }
        }
    }
    high_means.sort_by(f64::total_cmp);
    let median = high_means[high_means.len() / 2];
    eprintln!(
        "{parsed}/261 .aml parsed, {rebuilt}/261 rebuilt byte for byte; High vs the Rec.709 cube of the same look: {} looks, median of the mean abs diffs {median:.3e}, largest {:.3e}",
        high_means.len(),
        high_means.last().copied().unwrap_or(0.0)
    );
    assert_eq!((parsed, rebuilt), (261, 261));
    assert!(high_means.len() >= 80, "{}", high_means.len());
    // ARRI's own 3D-LUT version and the High look file are the same look to a fraction of a code.
    assert!(median < 1e-3, "{median:e}");
}

#[test]
#[ignore = "needs ARRI's real downloads: set YM_ARRI_DOWNLOADS (see the file header)"]
fn the_look_file_libraries_import_into_a_library_and_render() {
    let Some(source) = downloads() else {
        return;
    };
    for db in [shipped_db(), patterns_only()] {
        let by_hash = db.packages.iter().any(|p| p.is_fingerprinted());
        let work = tempfile::tempdir().unwrap();
        let mut ids_by_package: Vec<(&str, Vec<String>)> = Vec::new();
        for (file, name, package, files, _) in LOOK_FILE_ZIPS {
            let links = tempfile::tempdir().unwrap();
            let zip = linked(links.path(), &source.join(file), name);
            let mut library = open_library(&work.path().join(format!("app-{package}-{by_hash}")));
            let r = inspect(&zip, &db, None, &mut no_progress()).unwrap();
            let start = std::time::Instant::now();
            let report = import(
                &mut library,
                &db,
                &zip,
                &r,
                &ImportOptions::default(),
                &mut no_progress(),
            )
            .unwrap();
            eprintln!(
                "{} {package}: imported {} looks in {:?} ({} rejected)",
                if by_hash { "by hash" } else { "by names" },
                report.added.len(),
                start.elapsed(),
                report.rejected.len()
            );
            assert!(report.rejected.is_empty(), "{package}");
            assert_eq!(report.added.len(), files, "{package}");
            let mut moved = 0;
            for item in report.added.iter().step_by(7) {
                let look = library.get(&item.id).expect("look").clone();
                assert_eq!(
                    look.source.as_ref().and_then(|s| s.package.as_deref()),
                    Some(package)
                );
                let transform = library
                    .transform(&item.id)
                    .unwrap_or_else(|e| panic!("{}: {e}", item.id));
                for grey in [0.1, 0.2784, 0.5, 0.8] {
                    let input = LogC4([grey; 3]);
                    let out = if transform.includes_display_rendering() {
                        transform.eval_display_rec709(input).unwrap().0
                    } else {
                        transform.apply_logc4(input, 100.0).unwrap().0
                    };
                    assert!(out.iter().all(|v| v.is_finite()), "{}: {out:?}", item.id);
                    if (out[0] - grey).abs() > 1e-3 || (out[2] - grey).abs() > 1e-3 {
                        moved += 1;
                    }
                }
            }
            assert!(moved > 0, "{package}: no look changed anything");
            assert!(library.verify(&mut no_progress()).unwrap().is_clean());
            // A fresh library opens the same looks.
            let reopened = open_library(library.root());
            assert!(reopened.problems().is_empty());
            let mut ids: Vec<String> = reopened
                .looks()
                .iter()
                .filter(|l| l.id.prefix() == "arri")
                .map(|l| l.id.to_string())
                .collect();
            ids.sort();
            assert_eq!(ids.len(), files, "{package}");
            ids_by_package.push((package, ids));
        }
        // The ALF4 library has the ids of the 3D-LUT library: one look, one id, whichever was imported.
        let cube_zip = linked(
            tempfile::tempdir().unwrap().keep().as_path(),
            &source.join("logc4.zip"),
            LUT_ZIPS[0].1,
        );
        let mut cube_library = open_library(&work.path().join(format!("app-cubes-{by_hash}")));
        import_path(
            &mut cube_library,
            &db,
            &cube_zip,
            &ImportOptions::default(),
            &mut no_progress(),
        )
        .unwrap();
        let mut cube_ids: Vec<String> = cube_library
            .looks()
            .iter()
            .filter(|l| l.id.prefix() == "arri")
            .map(|l| l.id.to_string())
            .collect();
        cube_ids.sort();
        let alf4_ids = &ids_by_package[0].1;
        let same = alf4_ids.iter().filter(|id| cube_ids.contains(id)).count();
        eprintln!(
            "{}: {same} of {} .alf4 ids are also ids of the 3D-LUT library",
            if by_hash { "by hash" } else { "by names" },
            alf4_ids.len()
        );
        assert_eq!(
            same, 87,
            "the ALF4 library and the cube library name their looks alike"
        );
        assert_eq!(*alf4_ids, cube_ids);

        // Importing the look files into the library that has the cubes changes nothing; the other
        // way round, the cubes replace the look files.
        let alf_zip = linked(
            tempfile::tempdir().unwrap().keep().as_path(),
            &source.join("alf4.zip"),
            LOOK_FILE_ZIPS[0].1,
        );
        let report = import_path(
            &mut cube_library,
            &db,
            &alf_zip,
            &ImportOptions::default(),
            &mut no_progress(),
        )
        .unwrap()
        .1;
        assert!(report.added.is_empty() && report.updated.is_empty());
        assert_eq!(report.unchanged.len(), 87);
        eprintln!("cubes first, then look files: {:?}", report.notes);
        let mut alf_library = open_library(&work.path().join(format!("app-alf4-{by_hash}")));
        import_path(
            &mut alf_library,
            &db,
            &alf_zip,
            &ImportOptions::default(),
            &mut no_progress(),
        )
        .unwrap();
        let report = import_path(
            &mut alf_library,
            &db,
            &cube_zip,
            &ImportOptions::default(),
            &mut no_progress(),
        )
        .unwrap()
        .1;
        assert_eq!((report.added.len(), report.updated.len()), (0, 87));
        assert!(alf_library.verify(&mut no_progress()).unwrap().is_clean());
        eprintln!("look files first, then cubes: {:?}", report.notes);
    }
}

fn walk_first(dir: &Path, extensions: &[&str]) -> Option<PathBuf> {
    let mut entries: Vec<PathBuf> = fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            if path.file_name().is_some_and(|n| n == "__MACOSX") {
                continue;
            }
            if let Some(found) = walk_first(&path, extensions) {
                return Some(found);
            }
        } else if path
            .extension()
            .is_some_and(|e| extensions.iter().any(|x| e.eq_ignore_ascii_case(x)))
            && !path
                .file_name()
                .is_some_and(|n| n.to_string_lossy().starts_with("._"))
        {
            return Some(path);
        }
    }
    None
}

/// Imports a package into a fresh library, checks the library, and renders mid grey through a few
/// of its looks.
fn import_and_render(db: &PackageDb, zip: &Path, package: &str, library: &mut Library) -> usize {
    let r = inspect(zip, db, None, &mut no_progress()).unwrap();
    let start = std::time::Instant::now();
    let report = import(
        library,
        db,
        zip,
        &r,
        &ImportOptions::default(),
        &mut no_progress(),
    )
    .unwrap();
    eprintln!(
        "{package}: imported {} looks in {:?} ({} rejected)",
        report.added.len(),
        start.elapsed(),
        report.rejected.len()
    );
    assert!(report.rejected.is_empty());
    let added: Vec<_> = report.added.iter().collect();
    assert_eq!(added.len(), 87, "{package}");
    let mut changed = 0;
    for item in added.iter().step_by(10) {
        let look = library.get(&item.id).expect("look").clone();
        assert_eq!(
            look.source.as_ref().and_then(|s| s.package.as_deref()),
            Some(package)
        );
        let transform = library
            .transform(&item.id)
            .unwrap_or_else(|e| panic!("{}: {e}", item.id));
        for grey in [0.1, 0.2784, 0.5, 0.8] {
            let input = LogC4([grey; 3]);
            let out = if transform.includes_display_rendering() {
                transform.eval_display_rec709(input).unwrap().0
            } else {
                transform.apply_logc4(input, 100.0).unwrap().0
            };
            assert!(out.iter().all(|v| v.is_finite()), "{}: {out:?}", item.id);
            if (out[0] - grey).abs() > 1e-3 || (out[2] - grey).abs() > 1e-3 {
                changed += 1;
            }
        }
    }
    assert!(changed > 0, "{package}: no look changed anything");
    added.len()
}

#[test]
#[ignore = "needs ARRI's real downloads: set YM_ARRI_DOWNLOADS (see the file header)"]
fn the_real_packages_import_into_a_library_and_render() {
    let Some(source) = downloads() else {
        return;
    };
    let work = tempfile::tempdir().unwrap();
    let mut library = open_library(&work.path().join("app"));
    for db in [shipped_db(), patterns_only()] {
        for (file, name, package) in LUT_ZIPS {
            // Each pass starts from an empty library; the second pass proves names alone import too.
            let root = work.path().join(format!(
                "app-{package}-{}",
                db.packages[0].fingerprints.len()
            ));
            let mut lib = open_library(&root);
            let links = tempfile::tempdir().unwrap();
            let zip = linked(links.path(), &source.join(file), name);
            let n = import_and_render(&db, &zip, package, &mut lib);
            assert_eq!(n, 87);
            let arri = lib
                .looks()
                .iter()
                .filter(|l| l.id.prefix() == "arri")
                .count();
            assert_eq!(arri, 87, "{package}");
            assert!(lib.verify(&mut no_progress()).unwrap().is_clean());
            // Same ids and names for 'by hash' and 'by names': nothing depends on the fingerprints.
            let mut ids: Vec<String> = lib.looks().iter().map(|l| l.id.to_string()).collect();
            ids.sort();
            eprintln!("{package}: first ids {:?}", &ids[..4]);
            // The first library gets all three packages: their ids do not collide.
            if db.packages[0].is_fingerprinted() {
                import_and_render(&db, &zip, package, &mut library);
            }
        }
    }
    let arri = library
        .looks()
        .iter()
        .filter(|l| l.id.prefix() == "arri")
        .count();
    assert_eq!(
        arri,
        87 * 3,
        "the three libraries share names but never collide"
    );
    assert!(library.verify(&mut no_progress()).unwrap().is_clean());
}

/// The nine files that ARRI's look-builder wrote from our own synthetic cubes (`YM_ALF_SAMPLES`,
/// see `crates/ym-color/tests/alf_samples.rs`) through the importer: seven are looks, two are
/// refused in plain words (an `.alf4c` with no creative look, an `.aml` made for Rec.2020 PQ).
#[test]
#[ignore = "needs look files written by ART CMD: set YM_ALF_SAMPLES (see crates/ym-color/tests/alf_samples.rs)"]
fn the_look_builder_files_import_as_looks_or_are_refused_in_plain_words() {
    let Some(dir) = std::env::var_os("YM_ALF_SAMPLES").map(PathBuf::from) else {
        eprintln!("YM_ALF_SAMPLES is not set: nothing to check");
        return;
    };
    let folder = dir.join("out");
    let db = shipped_db();
    let work = tempfile::tempdir().unwrap();
    let mut library = open_library(&work.path().join("app"));
    let (r, report) = import_path(
        &mut library,
        &db,
        &folder,
        &ImportOptions::default(),
        &mut no_progress(),
    )
    .unwrap();
    assert_eq!(r.verdict(), Verdict::Ready, "{:?}", r.issues);
    assert_eq!(report.added.len(), 7, "{report:?}");
    assert_eq!(report.rejected.len(), 2, "{:?}", report.rejected);
    for rejected in &report.rejected {
        eprintln!("refused: {}: {}", rejected.name, rejected.reason);
    }
    let reason_of = |name: &str| {
        report
            .rejected
            .iter()
            .find(|f| f.name == name)
            .map(|f| f.reason.clone())
            .unwrap_or_else(|| panic!("{name} was not refused"))
    };
    assert!(reason_of("alf4c_nocmt.alf4c").contains("no creative look (CMT)"));
    assert_eq!(
        reason_of("aml_2020pq_17.aml"),
        "It is made for a Rec.2020 PQ display. Yellowmoon can use ARRI Look Files made for Rec.709."
    );
    // Each look renders, with the input its file says.
    for item in &report.added {
        let look = library.get(&item.id).unwrap();
        let transform = library.transform(&item.id).unwrap();
        let expected = if item.file.ends_with(".aml") {
            LookInput::LogC3ToRec709
        } else {
            LookInput::LogC4
        };
        assert_eq!(look.input, expected, "{}", item.file);
        assert_eq!(transform.input(), expected);
        eprintln!(
            "imported {} as {} ({}-point)",
            item.file,
            item.id,
            look.lut.as_ref().unwrap().lut_size
        );
    }
    // The .alf4c's creative look is its 17-point CMT, not a display transform.
    let alf4c = report
        .added
        .iter()
        .find(|i| i.file.ends_with("alf4c_cmt17_sdr33_hdr65.alf4c"))
        .unwrap();
    assert_eq!(
        library
            .get(&alf4c.id)
            .unwrap()
            .lut
            .as_ref()
            .unwrap()
            .lut_size,
        17
    );
    assert!(
        r.issues
            .iter()
            .any(|i| { i.kind == ym_looks::recognize::IssueKind::DisplayTransformsIgnored })
    );
    assert!(library.verify(&mut no_progress()).unwrap().is_clean());
}

/// The owner's realistic sequence of imports, on ARRI's real downloads, counting the library after
/// each step: nothing is ever listed twice, no look gets a collision number, and a step that adds
/// nothing leaves the list of looks byte for byte as it was (`tests/reimport.rs` has the same
/// sequence on made-up looks).
#[test]
#[ignore = "needs ARRI's real downloads: set YM_ARRI_DOWNLOADS (see the file header)"]
fn the_owners_sequence_of_imports_on_the_real_downloads_never_duplicates_a_look() {
    let Some(source) = downloads() else {
        return;
    };
    let zip_of = |links: &Path, file: &str| {
        let name = LUT_ZIPS
            .iter()
            .map(|z| (z.0, z.1))
            .chain(LOOK_FILE_ZIPS.iter().map(|z| (z.0, z.1)))
            .find(|z| z.0 == file)
            .expect("known download")
            .1;
        linked(links, &source.join(file), name)
    };
    let take = |library: &mut Library, db: &PackageDb, path: &Path, input| {
        let options = ImportOptions {
            input,
            ..ImportOptions::default()
        };
        import_path(library, db, path, &options, &mut no_progress())
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
            .1
    };
    let manifest = |library: &Library| fs::read(library.paths().manifest_path()).unwrap();
    let ids = |library: &Library| {
        let mut v: Vec<String> = library
            .looks()
            .iter()
            .filter(|l| l.id.prefix() != "builtin")
            .map(|l| l.id.to_string())
            .collect();
        v.sort();
        v
    };
    // After a step: this many looks (87 by ARRI and `user` by the person), every id once, a file
    // for each, a clean list.
    let check = |library: &Library, arri: usize, user: usize, label: &str| {
        let all = ids(library);
        let mut unique = all.clone();
        unique.dedup();
        assert_eq!(all, unique, "{label}: an id twice");
        let found_arri = all.iter().filter(|i| i.starts_with("arri:")).count();
        let found_user = all.iter().filter(|i| i.starts_with("user:")).count();
        assert_eq!((found_arri, found_user), (arri, user), "{label}");
        let files = fs::read_dir(library.looks_dir().join("arri"))
            .map(|d| {
                d.flatten()
                    .map(|g| fs::read_dir(g.path()).map_or(0, |f| f.flatten().count()))
                    .sum::<usize>()
            })
            .unwrap_or(0);
        assert_eq!(files, arri, "{label}: a file for each look");
        assert!(library.problems().is_empty(), "{label}");
        let mut copy = open_library(library.root());
        assert!(
            copy.verify(&mut no_progress()).unwrap().is_clean(),
            "{label}"
        );
        eprintln!("{label}: {arri} ARRI looks, {user} of your own");
    };

    for (pass, db) in [("by hash", shipped_db()), ("by names", patterns_only())] {
        eprintln!("--- {pass}");
        let work = tempfile::tempdir().unwrap();
        let links = tempfile::tempdir().unwrap();
        let logc4 = zip_of(links.path(), "logc4.zip");
        let logc3 = zip_of(links.path(), "logc3.zip");
        let rec709 = zip_of(links.path(), "logc3-rec709.zip");
        let alf4 = zip_of(links.path(), "alf4.zip");
        let alf2 = zip_of(links.path(), "alf2.zip");
        let mut library = open_library(&work.path().join("app"));

        // 1 and 2: the LogC4 3D LUTs, twice.
        let r = take(&mut library, &db, &logc4, None);
        assert_eq!(r.added.len(), 87);
        check(&library, 87, 0, "1 LogC4 3D LUTs");
        let logc4_ids = ids(&library);
        let list = manifest(&library);
        let r = take(&mut library, &db, &logc4, None);
        assert!(r.added.is_empty() && r.updated.is_empty(), "{r:?}");
        check(&library, 87, 0, "2 again");
        assert_eq!(manifest(&library), list, "2: the list is unchanged");

        // 3: the ALF4 zip.
        let r = take(&mut library, &db, &alf4, None);
        assert!(r.added.is_empty() && r.updated.is_empty(), "{r:?}");
        assert_eq!(r.unchanged.len(), 87);
        check(&library, 87, 0, "3 ALF4 zip");
        assert_eq!(ids(&library), logc4_ids);
        assert_eq!(manifest(&library), list, "3: the list is unchanged");

        // 4: both unzipped folders in one folder, one import. Names alone cannot tell the two
        // libraries apart inside one folder (no folder holds all the files); the hashes can, and
        // they are what the app ships.
        let both = work.path().join("both");
        let a = unzip_like_finder(
            &alf4,
            &both,
            "arri-look-library-logc4-log-to-log-alf4-data.zip",
        );
        let c = unzip_like_finder(
            &logc4,
            &both,
            "ARRI Look Library LogC4 (log-to-log) 3D-LUTs_data.zip",
        );
        assert!(a.is_dir() && c.is_dir());
        if pass == "by hash" {
            let r = take(&mut library, &db, &both, None);
            assert!(r.added.is_empty() && r.updated.is_empty(), "{r:?}");
            assert!(
                r.notes
                    .iter()
                    .any(|n| n.contains("came both as 3D LUTs and as look files")),
                "{:?}",
                r.notes
            );
            check(&library, 87, 0, "4 both in one import");
            assert_eq!(manifest(&library), list, "4: the list is unchanged");
        }

        // 5: the unzipped LogC4 folder.
        let r = take(&mut library, &db, &c, None);
        assert!(r.added.is_empty() && r.updated.is_empty(), "{r:?}");
        check(&library, 87, 0, "5 unzipped folder");
        assert_eq!(manifest(&library), list, "5: the list is unchanged");

        // 6 and 7: the two LogC3 libraries.
        let r = take(&mut library, &db, &logc3, None);
        assert_eq!(r.added.len(), 87, "{r:?}");
        check(&library, 174, 0, "6 LogC3");
        let r = take(&mut library, &db, &rec709, None);
        assert_eq!(r.added.len(), 87, "{r:?}");
        check(&library, 261, 0, "7 LogC3 to Rec709");

        // 8 and 9: the ALF2 zip (261 looks: three strengths of each), twice.
        let r = take(&mut library, &db, &alf2, None);
        assert_eq!(r.added.len(), 261, "{r:?}");
        check(&library, 522, 0, "8 ALF2");
        let list = manifest(&library);
        let r = take(&mut library, &db, &alf2, None);
        assert!(r.added.is_empty() && r.updated.is_empty(), "{r:?}");
        check(&library, 522, 0, "9 ALF2 again");
        assert_eq!(manifest(&library), list, "9: the list is unchanged");

        // 10 and 11: one cube of the person's own, and the same one renamed.
        let mine = work.path().join("mine");
        fs::create_dir_all(&mine).unwrap();
        let own = mine.join("Mine.cube");
        fs::write(&own, support::cube_text(17, 4242, "mine")).unwrap();
        let r = take(&mut library, &db, &own, Some(LookInput::LogC4));
        assert_eq!(r.added.len(), 1, "{r:?}");
        check(&library, 522, 1, "10 one cube of your own");
        let list = manifest(&library);
        let copy = mine.join("Renamed.cube");
        fs::copy(&own, &copy).unwrap();
        let r = take(&mut library, &db, &copy, Some(LookInput::LogC4));
        assert!(r.added.is_empty() && r.updated.is_empty(), "{r:?}");
        check(&library, 522, 1, "11 the same cube renamed");
        assert_eq!(manifest(&library), list, "11: the list is unchanged");

        // 12: the ALF4 zip into an empty library, then the 3D LUTs: they take the entries over.
        let mut fresh = open_library(&work.path().join("app-fresh"));
        let r = take(&mut fresh, &db, &alf4, None);
        assert_eq!(r.added.len(), 87, "{r:?}");
        check(&fresh, 87, 0, "12 ALF4 zip, empty library");
        let r = take(&mut fresh, &db, &logc4, None);
        assert!(r.added.is_empty(), "{r:?}");
        assert_eq!(r.updated.len(), 87, "{r:?}");
        check(&fresh, 87, 0, "12 then the 3D LUTs");
        assert_eq!(ids(&fresh), logc4_ids);
        assert!(
            fresh
                .looks()
                .iter()
                .filter(|l| l.id.prefix() == "arri")
                .all(|l| l.lut.as_ref().is_some_and(|u| u.format.is_cube())),
            "every look is the 3D LUT"
        );
    }
}

/// One file of a zip, written to `dir` under the name it has in the zip (loose, as a person drags
/// one file into the wizard).
fn extract_one(zip: &Path, tail: &str, dir: &Path) -> PathBuf {
    let mut archive = zip::ZipArchive::new(File::open(zip).expect("open")).expect("zip");
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i).expect("entry");
        let name = entry
            .enclosed_name()
            .map(|p| p.to_string_lossy().replace('\\', "/"))
            .unwrap_or_default();
        if name.starts_with("__MACOSX") || !name.ends_with(tail) {
            continue;
        }
        fs::create_dir_all(dir).expect("mkdir");
        let target = dir.join(tail);
        io::copy(&mut entry, &mut File::create(&target).expect("create")).expect("copy");
        return target;
    }
    panic!("{tail} is not in {}", zip.display());
}

/// A file that comes alone is the look its name says, with the id it has in the whole download: the
/// same look in its other form or in the whole package is found and never listed twice. Two files
/// of the LogC3 to Rec.709 package share their bytes (`5121-City` and `9111-Dream`): each is its
/// own look, whichever comes first.
#[test]
#[ignore = "needs ARRI's real downloads: set YM_ARRI_DOWNLOADS (see the file header)"]
fn single_files_of_the_real_downloads_are_the_looks_their_names_say() {
    let Some(source) = downloads() else {
        return;
    };
    let work = tempfile::tempdir().expect("work");
    let loose = work.path().join("loose");
    let calls = std::cell::Cell::new(0_u32);
    let zip_named = |file: &str| {
        let name = LUT_ZIPS
            .iter()
            .map(|z| (z.0, z.1))
            .chain(LOOK_FILE_ZIPS.iter().map(|z| (z.0, z.1)))
            .find(|z| z.0 == file)
            .expect("known download")
            .1;
        // A folder of its own each time: the link is made again for every import of the zip.
        let links = work.path().join(format!("links-{}", calls.get()));
        calls.set(calls.get() + 1);
        fs::create_dir_all(&links).expect("mkdir");
        linked(&links, &source.join(file), name)
    };
    let db = shipped_db();
    let take = |library: &mut Library, path: &Path| {
        import_path(
            library,
            &db,
            path,
            &ImportOptions::default(),
            &mut no_progress(),
        )
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .1
    };
    let arri_ids = |library: &Library| -> Vec<String> {
        let mut v: Vec<String> = library
            .looks()
            .iter()
            .filter(|l| l.id.prefix() == "arri")
            .map(|l| l.id.to_string())
            .collect();
        v.sort();
        v
    };
    let no_collision_numbers = |ids: &[String]| {
        for id in ids {
            let last = id.rsplit('-').next().unwrap_or("");
            assert!(
                !last.parse::<u32>().is_ok_and(|n| n < 100),
                "{id} carries a collision number"
            );
        }
    };

    // 1111 Black and White shares its title with 1110: alone it is already "... 1111".
    let cube = extract_one(
        &source.join("logc4.zip"),
        "1111 Black and White.cube",
        &loose,
    );
    let mut library = open_library(&work.path().join("app-cube-first"));
    let r = take(&mut library, &cube);
    assert_eq!(r.added.len(), 1, "{r:?}");
    assert_eq!(r.added[0].id.to_string(), "arri:1000:black-and-white-1111");
    assert_eq!(r.added[0].name, "Black and White 1111");
    // The whole download after it: 86 more looks, that one unchanged, no `-2`.
    let r = take(&mut library, &zip_named("logc4.zip"));
    assert_eq!((r.added.len(), r.unchanged.len()), (86, 1), "{r:?}");
    let ids = arri_ids(&library);
    assert_eq!(ids.len(), 87);
    no_collision_numbers(&ids);

    // The look file of the same look, alone, then the cubes: one entry, the cube wins.
    let look_file = extract_one(
        &source.join("alf4.zip"),
        "LL-1111-Black-and-White.alf4",
        &loose,
    );
    let mut library = open_library(&work.path().join("app-look-file-first"));
    let r = take(&mut library, &look_file);
    assert_eq!(r.added[0].id.to_string(), "arri:1000:black-and-white-1111");
    let r = take(&mut library, &zip_named("logc4.zip"));
    assert_eq!((r.added.len(), r.updated.len()), (86, 1), "{r:?}");
    assert_eq!(arri_ids(&library).len(), 87);
    // And the reverse: the cube first, then the look file alone changes nothing.
    let manifest = fs::read(library.paths().manifest_path()).unwrap();
    let r = take(&mut library, &look_file);
    assert!(r.added.is_empty() && r.updated.is_empty(), "{r:?}");
    assert_eq!(fs::read(library.paths().manifest_path()).unwrap(), manifest);

    // City and Dream hold the same LUT. Dream alone first, then the whole package: City is a look.
    let dream = extract_one(&source.join("logc3-rec709.zip"), "9111-Dream.cube", &loose);
    let mut library = open_library(&work.path().join("app-dream-first"));
    let r = take(&mut library, &dream);
    assert_eq!(r.added.len(), 1, "{r:?}");
    // Alone, it has the id it has in the whole package (its title is shared with 9110).
    assert_eq!(
        r.added[0].id.to_string(),
        "arri:9000:dream-9111-logc3-rec709"
    );
    let r = take(&mut library, &zip_named("logc3-rec709.zip"));
    assert_eq!(r.added.len(), 86, "{r:?}");
    let ids = arri_ids(&library);
    assert_eq!(ids.len(), 87, "{ids:?}");
    assert!(
        ids.contains(&"arri:5000:city-5121-logc3-rec709".to_string()),
        "{ids:?}"
    );
    no_collision_numbers(&ids);
    let mut again = open_library(library.root());
    assert!(again.verify(&mut no_progress()).unwrap().is_clean());
}
