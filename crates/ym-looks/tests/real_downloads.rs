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
//! downloads, and prints only names and counts.

use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};

use ym_color::logc::LogC4;
use ym_looks::import::{ImportOptions, import, inspect};
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

/// The camera look-file packages: file, name, and the extension the message must name.
const CAMERA_ZIPS: [(&str, &str, &str); 2] = [
    (
        "alf4.zip",
        "arri-look-library-logc4-log-to-log-alf4-data.zip",
        "(.alf4)",
    ),
    (
        "alf2.zip",
        "ARRI Look Library LogC3 (log-to-Rec709) ALF2_data.zip",
        "(.aml)",
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

#[test]
#[ignore = "needs ARRI's real downloads: set YM_ARRI_DOWNLOADS (see the file header)"]
fn the_camera_look_file_packages_say_which_download_to_take() {
    let Some(source) = downloads() else {
        return;
    };
    let db = shipped_db();
    for (file, name, extension) in CAMERA_ZIPS {
        let work = tempfile::tempdir().unwrap();
        let zip = source.join(file);
        let check = |label: &str, r: &Recognition| {
            assert_eq!(r.verdict(), Verdict::Blocked, "{label}");
            assert!(r.files.is_empty() && r.packages.is_empty(), "{label}");
            let message = &r.issues[0].message;
            assert!(
                message.starts_with("These are ARRI look files for the camera "),
                "{label}: {message}"
            );
            assert!(message.contains(extension), "{label}: {message}");
            assert!(message.contains("3D-LUTs"), "{label}: {message}");
            eprintln!("{label}: {message}");
        };
        let r = inspect(
            &linked(work.path(), &zip, name),
            &db,
            None,
            &mut no_progress(),
        )
        .unwrap();
        check(name, &r);
        assert_eq!(
            r.ignored.alf4 + r.ignored.aml,
            if file == "alf4.zip" { 87 } else { 261 }
        );
        let outer = unzip_like_finder(&zip, work.path(), name);
        check(
            "folder",
            &inspect(&outer, &db, None, &mut no_progress()).unwrap(),
        );
        check(
            "inner folder",
            &inspect(&inner_folder(&outer), &db, None, &mut no_progress()).unwrap(),
        );
        // One look file chosen by itself.
        let loose = walk_first(&outer, &["alf4", "alf4c", "aml"]).expect("a look file");
        check(
            "loose file",
            &inspect(&loose, &db, None, &mut no_progress()).unwrap(),
        );
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
