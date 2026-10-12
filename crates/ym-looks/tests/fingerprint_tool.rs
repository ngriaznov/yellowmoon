//! Cross-check of `tools/looks/fingerprint.py` against the Rust recogniser: what the script writes
//! must be what `ym-looks` reads, and must identify a package exactly, whatever its files are
//! called. Skipped (with a message) when `python3` is not installed.

mod support;

use std::fs;
use std::path::Path;
use std::process::Command;

use support::{Kind, no_progress, open_library, package_files, write_package_zip, write_zip};
use ym_looks::import::{ImportOptions, import, inspect};
use ym_looks::recognize::{Evidence, Outcome, PackageDb, Verdict};

fn python() -> Option<&'static str> {
    ["python3", "python"].into_iter().find(|p| {
        Command::new(p)
            .arg("--version")
            .output()
            .is_ok_and(|o| o.status.success())
    })
}

#[test]
fn the_script_output_identifies_a_renamed_package_by_hash() {
    let Some(python) = python() else {
        eprintln!("python3 not found: skipping the fingerprint.py cross-check");
        return;
    };
    let tool = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/looks/fingerprint.py");
    let shipped = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/looks/arri-packages.json");
    let work = tempfile::tempdir().unwrap();
    let list = work.path().join("arri-packages.json");
    fs::copy(&shipped, &list).unwrap();

    // 1. Fingerprint a synthetic LogC4 package (4 small cubes) with the real script.
    let zip = write_package_zip(work.path(), Kind::LookLogC4, 4);
    let out = Command::new(python)
        .arg("-I")
        .arg(&tool)
        .arg(&zip)
        .arg("--packages")
        .arg(&list)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    // 2. The Rust side reads the file the script wrote.
    let db = PackageDb::load(&list).unwrap();
    let package = db.package("look-library-logc4").unwrap();
    assert_eq!(package.fingerprints.len(), 4);
    assert!(
        db.package("look-library-logc3")
            .unwrap()
            .fingerprints
            .is_empty()
    );
    assert_eq!(package.expected.cube_files, Some(4));
    assert_eq!(package.expected.lut_sizes, vec![5]);
    let texts = package_files(Kind::LookLogC4, 4);
    for (path, text) in &texts {
        let fp = package
            .fingerprints
            .iter()
            .find(|f| &f.path == path)
            .unwrap();
        assert_eq!(fp.size, text.len() as u64);
        assert_eq!(fp.lut_size, 5);
        assert_eq!(fp.sha256, ym_looks::manifest::sha256_hex(text.as_bytes()));
    }

    // 3. A zip of the same files with meaningless names is identified exactly.
    let renamed: Vec<(String, Vec<u8>)> = texts
        .iter()
        .enumerate()
        .map(|(i, (_, t))| (format!("file{i}.cube"), t.clone().into_bytes()))
        .collect();
    let anonymous = work.path().join("download.zip");
    write_zip(&anonymous, &renamed);
    let recognition = inspect(&anonymous, &db, None, &mut no_progress()).unwrap();
    assert_eq!(
        recognition.verdict(),
        Verdict::Ready,
        "{:?}",
        recognition.issues
    );
    assert_eq!(recognition.packages[0].evidence, Evidence::Fingerprint);
    assert!(recognition.files.iter().all(|f| matches!(
        &f.outcome,
        Outcome::ArriLook { package, evidence: Evidence::Fingerprint, .. } if package == "look-library-logc4"
    )));
    let mut library = open_library(&work.path().join("app"));
    let report = import(
        &mut library,
        &db,
        &anonymous,
        &recognition,
        &ImportOptions::default(),
        &mut no_progress(),
    )
    .unwrap();
    assert_eq!(report.added.len(), 4);
    assert!(
        library
            .get(&ym_looks::library::LookId::arri(3000, "look-02"))
            .is_some()
    );
}
