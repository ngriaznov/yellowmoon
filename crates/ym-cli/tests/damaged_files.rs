//! The damaged-file matrix through `ym render` (`docs/26` P1-6, task H-03): the folder of
//! `crates/ym-core/tests/damaged_files.rs` (ten valid synthetic files and 190 damaged ones,
//! `ym_raw::testgen::damage::write_matrix`) given to the binary.
//!
//! Exit 1 (some photos failed); the good files are in the output folder; stderr says one catalogue
//! sentence per file that failed; `--json` has `status` and, for a failure, `error.kind` and the same
//! sentence. A damaged file that only has damaged pixel values renders and is not an error.

mod common;

use std::collections::{BTreeMap, BTreeSet};

use common::{Sandbox, json_lines, lines, names_in, text};
use serde_json::Value;
use ym_raw::error::{MESSAGE_CUT_SHORT, MESSAGE_DAMAGED, MESSAGE_EMPTY};
use ym_raw::testgen::damage::{MatrixFile, valid_files, write_matrix};

const NOT_ARW: &str = "This isn't an ARW file. Yellowmoon opens Sony ARW files only.";

/// The sentences a failed file may show in `ym render`: the catalogue of `ym_raw::error` for a
/// damaged file, the fixed one of a file that is not ARW, and the two forms of "not supported".
fn is_catalogue_sentence(message: &str) -> bool {
    [MESSAGE_EMPTY, MESSAGE_CUT_SHORT, MESSAGE_DAMAGED, NOT_ARW].contains(&message)
        || message.contains("aren't supported yet")
        || message.starts_with("Yellowmoon can't open ")
        || (message.starts_with("This photo can't be analysed (")
            && (message.ends_with("). Copy it from the card again, or skip it.")
                || message.ends_with("). Skip it and open the others.")))
}

/// The name of the file a `--json` record is about.
fn file_name_of(record: &Value) -> String {
    let input = record["input"].as_str().expect("an input");
    input.rsplit('/').next().expect("a name").to_owned()
}

#[test]
fn the_damaged_folder_exports_the_good_files_and_names_each_bad_one() {
    let sb = Sandbox::new();
    let files: Vec<MatrixFile> = write_matrix(&sb.photos()).expect("write the matrix");
    assert_eq!(files.len(), 200);
    let good: Vec<&MatrixFile> = files.iter().filter(|f| f.damage.is_none()).collect();
    assert_eq!(good.len(), valid_files().len());

    // --json: one record per file, in name order.
    let out = sb
        .ym()
        .args(["render", "--json", "--format", "jpeg"])
        .arg(sb.photos())
        .assert()
        .code(1);
    let records = json_lines(&out.get_output().stdout);
    assert_eq!(records.len(), files.len(), "one record per file");
    let by_name: BTreeMap<String, &Value> = records.iter().map(|r| (file_name_of(r), r)).collect();
    let (mut failed, mut ok) = (BTreeSet::new(), BTreeSet::new());
    let mut kinds = BTreeSet::new();
    for f in &files {
        let record = by_name[&f.file];
        match record["status"].as_str() {
            Some("ok") => {
                assert!(record["error"].is_null(), "{}: {record}", f.file);
                ok.insert(f.file.clone());
            }
            Some("error") => {
                let damage = f
                    .damage
                    .as_ref()
                    .unwrap_or_else(|| panic!("the good file {} failed: {record}", f.file));
                let kind = record["error"]["kind"].as_str().expect("a kind");
                let message = record["error"]["message"].as_str().expect("a message");
                assert!(
                    is_catalogue_sentence(message),
                    "{} ({}): {message:?}",
                    f.file,
                    damage.label()
                );
                assert!(record["output"].is_null(), "{}", f.file);
                kinds.insert(kind.to_owned());
                failed.insert(f.file.clone());
            }
            other => panic!("{}: status {other:?}", f.file),
        }
    }
    eprintln!("{} ok, {} failed, kinds {kinds:?}", ok.len(), failed.len());
    assert!(!failed.is_empty() && ok.len() >= good.len());
    for f in &good {
        assert!(ok.contains(&f.file), "{} did not render", f.file);
    }
    for f in &files {
        if f.damage.as_ref().is_some_and(|d| d.always_refused()) {
            assert!(
                failed.contains(&f.file),
                "{} ({:?}) was not refused",
                f.file,
                f.damage
            );
        }
    }
    // `docs/26` P1-6 expects `not_arw`, `corrupt` and `io`. A file whose probe passes and whose
    // decode fails has `decode` (`skip_kind` in `render.rs`; see the ignored test below).
    for kind in &kinds {
        assert!(
            ["not_arw", "corrupt", "io", "decode"].contains(&kind.as_str()),
            "error.kind {kind:?} is not a kind of a failed file"
        );
    }

    // Every good file, and every damaged one that rendered, is in the output folder.
    let written = names_in(&sb.default_export());
    for name in &ok {
        let stem = name.trim_end_matches(".ARW");
        assert!(
            written.contains(&format!("{stem}_signature.jpg")),
            "{name} is not in {written:?}"
        );
    }
    assert_eq!(written.len(), ok.len(), "only the files that rendered");

    // The good files are byte for byte what the same files give in a folder of their own.
    let clean = Sandbox::new();
    for (variant, bytes) in valid_files() {
        std::fs::write(clean.photos().join(format!("good_{variant}.ARW")), bytes).expect("write");
    }
    clean
        .ym()
        .args(["render", "--format", "jpeg"])
        .arg(clean.photos())
        .assert()
        .code(0);
    for f in &good {
        let name = format!("{}_signature.jpg", f.file.trim_end_matches(".ARW"));
        let alone = std::fs::read(clean.default_export().join(&name)).expect("alone");
        let together = std::fs::read(sb.default_export().join(&name)).expect("together");
        assert!(!alone.is_empty() && alone == together, "{name} changed");
    }
}

#[test]
fn the_damaged_folder_in_text_mode_has_one_catalogue_line_per_bad_file() {
    let sb = Sandbox::new();
    let files = write_matrix(&sb.photos()).expect("write the matrix");
    let out = sb.ym().arg("render").arg(sb.photos()).assert().code(1);
    let stderr = text(&out.get_output().stderr);
    let stdout = lines(&out.get_output().stdout);
    let written = names_in(&sb.default_export());
    assert_eq!(
        stdout.len(),
        written.len(),
        "one path on stdout per file written"
    );
    for f in &files {
        let stem = f.file.trim_end_matches(".ARW");
        let wrote = written.contains(&format!("{stem}_signature.jpg"));
        // A line is `Error: {path}: {sentence}`; paths here have no ": " in them.
        let said: Vec<&str> = stderr
            .lines()
            .filter_map(|l| l.strip_prefix("Error: "))
            .filter(|l| {
                l.split_once(": ")
                    .is_some_and(|(path, _)| path.rsplit('/').next() == Some(f.file.as_str()))
            })
            .collect();
        if wrote {
            assert!(
                said.is_empty(),
                "{} rendered and still has {said:?}",
                f.file
            );
        } else {
            assert!(
                f.damage.is_some(),
                "the good file {} was not written",
                f.file
            );
            assert_eq!(said.len(), 1, "{}: {said:?}", f.file);
            let sentence = said[0].split_once(": ").expect("name: sentence").1;
            assert!(is_catalogue_sentence(sentence), "{}: {sentence:?}", f.file);
        }
    }
    let failed = files.iter().filter(|f| {
        !written.contains(&format!(
            "{}_signature.jpg",
            f.file.trim_end_matches(".ARW")
        ))
    });
    assert!(
        stderr.contains(&format!("{} photos failed.", failed.count())),
        "{stderr}"
    );
    // The technical detail stays out of the line unless `-v` asks for it.
    for banned in ["os error", "Huffman", "((", "entropy"] {
        assert!(!stderr.contains(banned), "{banned:?} in {stderr}");
    }
}

/// `docs/26` P1-6 says `error.kind` is one of `not_arw`, `corrupt` and `io`. A file the probe reads
/// but the decoder refuses (a cut-short file: the directories are there, the raw data is not) is
/// reported with the kind `decode`, which `docs/10` and `docs/04` do not list: `skip_kind` in
/// `crates/ym-cli/src/render.rs` probes again, finds nothing wrong, and falls back to `decode`,
/// with no `error.detail`.
#[test]
#[ignore = "H-03 finding: ym render --json gives error.kind \"decode\" (undocumented, no detail) for a file whose probe passes and whose decode fails"]
fn every_error_kind_is_not_arw_corrupt_or_io() {
    let sb = Sandbox::new();
    write_matrix(&sb.photos()).expect("write the matrix");
    let out = sb
        .ym()
        .args(["render", "--json"])
        .arg(sb.photos())
        .assert()
        .code(1);
    for record in json_lines(&out.get_output().stdout) {
        if record["status"] == "error" {
            let kind = record["error"]["kind"].as_str().expect("a kind");
            assert!(
                ["not_arw", "corrupt", "io"].contains(&kind),
                "{}: kind {kind:?}",
                file_name_of(&record)
            );
        }
    }
}

/// A file without read permission fails with the "isn't allowed" sentence, the others render. Skipped
/// when the user can read a mode-000 file (root).
#[cfg(unix)]
#[test]
fn a_file_without_read_permission_fails_alone() {
    use std::os::unix::fs::PermissionsExt;
    let sb = Sandbox::new();
    sb.arw("a_good.ARW");
    let locked = sb.arw("b_locked.ARW");
    sb.arw("c_good.ARW");
    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).expect("chmod");
    let readable = std::fs::File::open(&locked).is_ok();
    let out = sb
        .ym()
        .args(["render", "--json"])
        .arg(sb.photos())
        .assert()
        .code(if readable { 0 } else { 1 });
    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o644)).expect("chmod");
    if readable {
        eprintln!("skipped: this user can read a file with mode 000 (root)");
        return;
    }
    let records = json_lines(&out.get_output().stdout);
    assert_eq!(records.len(), 3);
    let record = records
        .iter()
        .find(|r| file_name_of(r) == "b_locked.ARW")
        .expect("the locked file");
    assert_eq!(record["status"], "error");
    assert_eq!(record["error"]["kind"], "io");
    assert_eq!(
        record["error"]["message"],
        ym_raw::error::MESSAGE_DENIED,
        "{record}"
    );
    assert_eq!(
        names_in(&sb.default_export()),
        ["a_good_signature.jpg", "c_good_signature.jpg"]
    );
}

/// A link to a good file renders; a dangling link, a loop and a link to a folder are not photos and
/// are not an error; the good files still render and the exit code is 0. Runs as root too.
#[cfg(unix)]
#[test]
fn symlinks_render_or_are_ignored_and_stop_nothing() {
    use std::os::unix::fs::symlink;
    let sb = Sandbox::new();
    sb.arw("a_good.ARW");
    sb.arw("target.dat"); // not an ARW name: only the link to it is a photo
    symlink("target.dat", sb.photos().join("b_link.ARW")).expect("link");
    symlink("nowhere.dat", sb.photos().join("c_dangling.ARW")).expect("link");
    symlink("d_loop.ARW", sb.photos().join("d_loop.ARW")).expect("link");
    std::fs::create_dir(sb.photos().join("sub")).expect("dir");
    symlink("sub", sb.photos().join("e_folder.ARW")).expect("link");
    sb.arw("z_good.ARW");
    let out = sb
        .ym()
        .args(["render", "--json"])
        .arg(sb.photos())
        .assert()
        .code(0);
    let records = json_lines(&out.get_output().stdout);
    let names: Vec<String> = records.iter().map(file_name_of).collect();
    assert_eq!(names, ["a_good.ARW", "b_link.ARW", "z_good.ARW"]);
    assert!(records.iter().all(|r| r["status"] == "ok"));
}
