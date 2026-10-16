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
