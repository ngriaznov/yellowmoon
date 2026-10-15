//! Move-to-Trash. The real Trash is never touched: on Linux the test runs the move in a child
//! process whose `XDG_DATA_HOME` and `HOME` point into a temporary folder, then looks inside that
//! folder's `Trash`. On other platforms only the "missing file" case runs, because the system
//! Trash cannot be redirected there.

use std::env;
#[cfg(target_os = "linux")]
use std::fs; // only the Linux-only test below touches files; unused (and a clippy error) elsewhere
use std::path::Path;

use ym_looks::import::trash::{TrashError, move_to_trash};

/// The child half: moves the file named by the environment to the (redirected) Trash.
#[test]
fn child_trash() {
    let Ok(file) = env::var("YM_LOOKS_TRASH_FILE") else {
        return;
    };
    move_to_trash(Path::new(&file)).unwrap_or_else(|e| panic!("{e}"));
}

#[test]
fn moving_a_missing_file_says_so() {
    let dir = tempfile::tempdir().unwrap();
    let err = move_to_trash(&dir.path().join("gone.zip")).unwrap_err();
    assert!(matches!(err, TrashError::Missing(_)), "{err}");
    assert!(err.to_string().contains("gone.zip"));
}

#[cfg(target_os = "linux")]
#[test]
fn the_zip_lands_in_the_trash_and_can_be_put_back() {
    let sandbox = tempfile::tempdir().unwrap();
    let home = sandbox.path().join("home");
    let data = sandbox.path().join("data");
    fs::create_dir_all(&home).unwrap();
    fs::create_dir_all(&data).unwrap();
    // Same volume as the redirected data folder, so the freedesktop home Trash is used.
    let zip = sandbox.path().join("ARRI_Look_Library_LogC4.zip");
    fs::write(&zip, b"zip bytes").unwrap();

    let out = std::process::Command::new(env::current_exe().unwrap())
        .args(["child_trash", "--exact", "--nocapture", "--test-threads=1"])
        .env("HOME", &home)
        .env("XDG_DATA_HOME", &data)
        .env("YM_LOOKS_TRASH_FILE", &zip)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(!zip.exists(), "the zip should be gone from its folder");
    let trashed = data.join("Trash/files/ARRI_Look_Library_LogC4.zip");
    assert_eq!(
        fs::read(trashed).unwrap(),
        b"zip bytes",
        "recoverable from the Trash"
    );
    let info =
        fs::read_to_string(data.join("Trash/info/ARRI_Look_Library_LogC4.zip.trashinfo")).unwrap();
    assert!(info.contains("[Trash Info]"));
}
