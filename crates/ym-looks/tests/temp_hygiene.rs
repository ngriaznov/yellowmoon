//! Loading the library removes the temporary files of a crashed import that are older than 24 hours
//! (`docs/26-hardening.md` P2-4, task H-08): `.<name>.tmp-<pid>-<n>` in `looks/` and its group folders.

use std::fs::{self, OpenOptions};
use std::path::Path;
use std::time::{Duration, SystemTime};

use ym_looks::library::Library;
use ym_looks::manifest::{STALE_TEMP_AGE, is_temp_name, remove_stale_temps};

fn file(path: &Path, hours: u64) {
    fs::create_dir_all(path.parent().expect("parent")).expect("dir");
    fs::write(path, b"half a look").expect("write");
    OpenOptions::new()
        .write(true)
        .open(path)
        .expect("open")
        .set_modified(SystemTime::now() - Duration::from_secs(hours * 3600))
        .expect("mtime");
}

#[test]
fn names() {
    assert!(is_temp_name(".manifest.json.tmp-123-0"));
    assert!(is_temp_name(".Look A.cube.tmp-1-22"));
    for no in [
        "manifest.json.tmp-1-2",
        ".manifest.json.tmp-1",
        ".manifest.json.tmp-a-2",
        ".manifest.json.tmp-1-2.bak",
        ".tmp-1-2",
        ".manifest.json",
        "manifest.json.damaged-1",
    ] {
        assert!(!is_temp_name(no), "{no}");
    }
}

#[test]
fn opening_the_library_removes_old_temporaries_and_keeps_the_rest() {
    let root = tempfile::tempdir().expect("root");
    let looks = root.path().join("looks");
    file(&looks.join(".manifest.json.tmp-77-1"), 25);
    file(&looks.join(".manifest.json.tmp-77-2"), 23);
    file(&looks.join("Group").join(".A.cube.tmp-77-3"), 26);
    file(&looks.join("Group").join(".B.cube.tmp-77-4"), 1);
    file(&looks.join("Group").join("C.cube"), 500);
    file(&looks.join(".manifest.json.tmp-77-5.keep"), 500);

    let _library = Library::open(root.path(), None).expect("opens");
    let mut left: Vec<String> = Vec::new();
    for dir in [looks.clone(), looks.join("Group")] {
        for e in fs::read_dir(dir).expect("dir").flatten() {
            if e.path().is_file() {
                left.push(e.file_name().to_string_lossy().into_owned());
            }
        }
    }
    left.sort();
    assert_eq!(
        left,
        [
            ".B.cube.tmp-77-4",
            ".manifest.json.tmp-77-2",
            ".manifest.json.tmp-77-5.keep",
            "C.cube"
        ]
    );
}

#[test]
fn a_library_folder_that_is_not_there_is_not_made() {
    let root = tempfile::tempdir().expect("root");
    let _library = Library::open(root.path(), None).expect("opens");
    assert!(!root.path().join("looks").exists());
    assert_eq!(
        remove_stale_temps(
            &root.path().join("looks"),
            SystemTime::now(),
            STALE_TEMP_AGE
        ),
        0
    );
}
