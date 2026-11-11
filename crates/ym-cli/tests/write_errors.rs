//! A destination that cannot be written, through `ym render` (`docs/26` P2-1, task H-05): the
//! `error.kind` (`disk_full`, `destination` or `write`), the sentence on stderr and in `--json`, the
//! full path of the folder, and the operating system's text in `error.detail` only.
//!
//! The classes that need a mount (a read-only volume, a full disk) run when the test may mount a
//! small `tmpfs` (Linux, usually root) and say so when it may not. The others use paths the kernel
//! refuses: `/proc` has no new folders (a folder that is not there) and `/sys` forbids them.

mod common;

use std::path::{Path, PathBuf};
use std::process::Command;

use common::{Sandbox, json_lines, lines, names_in, text};
use serde_json::Value;
use ym_export::DestinationProblem;

/// Runs `ym render --json` of one photo to `out`; returns its record.
fn render_to(sb: &Sandbox, out: &Path, extra: &[&str]) -> Value {
    let photo = sb.arw("DSC00001.ARW");
    let run = sb
        .ym()
        .args(["render", "--json", "--format", "png"])
        .args(extra)
        .arg("--out")
        .arg(out)
        .arg(&photo)
        .assert()
        .code(1);
    let output = run.get_output();
    let records = json_lines(&output.stdout);
    assert_eq!(records.len(), 1, "{records:?}");
    records[0].clone()
}

/// The same run without `--json`: stderr has one line, the input and the sentence.
fn stderr_of(sb: &Sandbox, out: &Path, extra: &[&str]) -> String {
    let photo = sb.arw("DSC00002.ARW");
    let run = sb
        .ym()
        .args(["render", "--format", "png"])
        .args(extra)
        .arg("--out")
        .arg(out)
        .arg(&photo)
        .assert()
        .code(1);
    text(&run.get_output().stderr)
}

fn assert_no_os_text(message: &str) {
    for banned in ["os error", "(os error", "Os {"] {
        assert!(!message.contains(banned), "{message}");
    }
}

#[test]
fn a_folder_that_is_not_there_is_a_destination_problem() {
    if !cfg!(target_os = "linux") {
        eprintln!("skipped: needs /proc");
        return;
    }
    let sb = Sandbox::new();
    let out = Path::new("/proc/ym-no-such-folder/exports");
    // What the kernel says to the same request decides what the class must be.
    let Err(e) = std::fs::create_dir_all(out) else {
        eprintln!("skipped: /proc accepts new folders here");
        return;
    };
    let want = DestinationProblem::of_io(&e).expect("a destination problem");
    assert_eq!(want, DestinationProblem::Gone, "{e}");
    let record = render_to(&sb, out, &[]);
    assert_eq!(record["status"], "error");
    assert_eq!(record["error"]["kind"], "destination");
    let message = record["error"]["message"].as_str().expect("a message");
    assert_eq!(
        message,
        "/proc/ym-no-such-folder/exports is not there any more. Check the path or the drive, then run the command again."
    );
    assert_no_os_text(message);
    let detail = record["error"]["detail"].as_str().expect("a detail");
    assert!(detail.contains("No such file"), "{detail}");
    let stderr = stderr_of(&sb, out, &[]);
    assert!(stderr.contains(message), "{stderr}");
    assert_no_os_text(&stderr);
}

#[test]
fn a_folder_that_may_not_be_written_is_a_permission_problem() {
    if !cfg!(target_os = "linux") {
        eprintln!("skipped: needs /sys");
        return;
    }
    let sb = Sandbox::new();
    let out = Path::new("/sys/ym-no-permission/exports");
    let Err(e) = std::fs::create_dir_all(out) else {
        eprintln!("skipped: /sys accepts new folders here");
        return;
    };
    if DestinationProblem::of_io(&e) != Some(DestinationProblem::Permission) {
        eprintln!("skipped: /sys answers {e} here");
        return;
    }
    let record = render_to(&sb, out, &[]);
    assert_eq!(record["error"]["kind"], "destination");
    let message = record["error"]["message"].as_str().expect("a message");
    assert_eq!(
        message,
        "Yellowmoon isn't allowed to write to /sys/ym-no-permission/exports. Choose another folder with --out."
    );
    assert!(record["error"]["detail"].is_string());
    let stderr = stderr_of(&sb, out, &[]);
    assert!(stderr.contains(message), "{stderr}");
    assert_no_os_text(&stderr);
}

#[test]
fn another_write_failure_is_kind_write_with_the_generic_sentence() {
    let sb = Sandbox::new();
    // A file where the folder should be: the folder cannot be made, and that says nothing about
    // the destination as a whole.
    let blocker = sb.root().join("blocker");
    std::fs::write(&blocker, b"x").expect("blocker");
    let out = blocker.join("exports");
    let record = render_to(&sb, &out, &[]);
    assert_eq!(record["error"]["kind"], "write");
    let message = record["error"]["message"].as_str().expect("a message");
    assert_eq!(
        message,
        format!(
            "Couldn't write to {}. Check that the folder is still there and that you can save to it.",
            out.display()
        )
    );
    assert_no_os_text(message);
    assert!(record["error"]["detail"].is_string(), "{record}");
    let stderr = stderr_of(&sb, &out, &[]);
    assert!(stderr.contains(message), "{stderr}");
}

/// A small `tmpfs` mounted for one test, unmounted when dropped.
struct Mount(PathBuf);

impl Mount {
    fn new(at: &Path, size: &str, read_only: bool) -> Option<Self> {
        std::fs::create_dir_all(at).ok()?;
        let ok = |cmd: &mut Command| cmd.status().is_ok_and(|s| s.success());
        if !ok(Command::new("mount")
            .args(["-t", "tmpfs", "-o"])
            .arg(format!("size={size}"))
            .arg("tmpfs")
            .arg(at))
        {
            return None;
        }
        let mount = Self(at.to_path_buf());
        if read_only && !ok(Command::new("mount").args(["-o", "remount,ro"]).arg(at)) {
            return None;
        }
        Some(mount)
    }
}

impl Drop for Mount {
    fn drop(&mut self) {
        let _ = Command::new("umount").arg(&self.0).status();
    }
}

#[test]
fn a_read_only_volume_is_a_destination_problem() {
    let sb = Sandbox::new();
    let at = sb.root().join("ro");
    let Some(_mount) = Mount::new(&at, "64k", true) else {
        eprintln!("skipped: this user cannot mount a tmpfs");
        return;
    };
    let record = render_to(&sb, &at, &[]);
    assert_eq!(record["error"]["kind"], "destination");
    let message = record["error"]["message"].as_str().expect("a message");
    assert_eq!(
        message,
        format!(
            "{} is read-only. Choose another folder with --out.",
            at.display()
        )
    );
    assert!(
        record["error"]["detail"]
            .as_str()
            .is_some_and(|d| d.contains("Read-only"))
    );
    let stderr = stderr_of(&sb, &at, &[]);
    assert!(stderr.contains(message), "{stderr}");
    assert_no_os_text(&stderr);
}

#[test]
fn a_full_disk_is_kind_disk_full_and_leaves_nothing_behind() {
    let sb = Sandbox::new();
    let at = sb.root().join("small");
    let Some(_mount) = Mount::new(&at, "32k", false) else {
        eprintln!("skipped: this user cannot mount a tmpfs");
        return;
    };
    let record = render_to(&sb, &at, &["--bits", "16"]);
    assert_eq!(record["error"]["kind"], "disk_full");
    let message = record["error"]["message"].as_str().expect("a message");
    assert_eq!(
        message,
        format!(
            "Stopped: the disk holding {} is full. Free some space, then run the command again with --skip-existing.",
            at.display()
        )
    );
    assert!(
        record["error"]["detail"]
            .as_str()
            .is_some_and(|d| d.contains("No space left"))
    );
    let stderr = stderr_of(&sb, &at, &["--bits", "16"]);
    assert!(stderr.contains(message), "{stderr}");
    assert_no_os_text(&stderr);
    assert!(
        lines(stderr.as_bytes())
            .iter()
            .all(|l| !l.contains(".yellowmoon-")),
        "{stderr}"
    );
    assert!(names_in(&at).is_empty(), "{:?}", names_in(&at));
}
