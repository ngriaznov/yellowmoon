//! `ym render` warns before a run when the destination is short of room (`docs/26` P2-3, task H-07):
//! "This export needs about {size} and {free} is free in {folder}. It may stop part way." on stderr, and
//! with `--json` a `low_space` event on stdout. It never stops the run: the exact stop on a full disk is
//! H-06's.
//!
//! The small disk is a `tmpfs`, which needs `mount` (Linux, usually root); the tests say so and return
//! when it is refused.

mod common;

use std::path::{Path, PathBuf};
use std::process::Command;

use common::{Sandbox, json_lines, lines, text};

/// A small `tmpfs` mounted for one test, unmounted when dropped.
struct Mount(PathBuf);

impl Mount {
    fn new(at: &Path, size_kib: u64) -> Option<Self> {
        std::fs::create_dir_all(at).ok()?;
        let status = Command::new("mount")
            .args(["-t", "tmpfs", "-o"])
            .arg(format!("size={size_kib}k"))
            .arg("tmpfs")
            .arg(at)
            .status()
            .ok()?;
        status.success().then(|| Self(at.to_path_buf()))
    }
}

impl Drop for Mount {
    fn drop(&mut self) {
        let _ = Command::new("umount").arg(&self.0).status();
    }
}

fn photos(sb: &Sandbox, n: usize) -> Vec<PathBuf> {
    (0..n)
        .map(|k| sb.arw_sized(&format!("DSC{:05}.ARW", k + 1), 160, 120, 11 + k as u64))
        .collect()
}

fn render(sb: &Sandbox, photos: &[PathBuf], out: &Path, json: bool) -> std::process::Output {
    let mut cmd = sb.ym();
    cmd.args(["render", "--format", "png", "--bits", "16"]);
    if json {
        cmd.arg("--json");
    }
    cmd.arg("--out").arg(out).args(photos);
    cmd.output().expect("ym runs")
}

fn warning(stderr: &str) -> Option<String> {
    lines(stderr.as_bytes())
        .into_iter()
        .find(|l| l.contains("This export needs about"))
}

#[test]
fn a_destination_with_room_gets_no_warning() {
    let sb = Sandbox::new();
    let photos = photos(&sb, 2);
    let out_dir = sb.root().join("roomy");
    let out = render(&sb, &photos, &out_dir, false);
    let stderr = text(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "{stderr}");
    assert!(warning(&stderr).is_none(), "{stderr}");
}

#[test]
fn a_small_disk_gets_the_warning_and_the_run_still_goes_on() {
    let sb = Sandbox::new();
    let photos = photos(&sb, 3);
    let at = sb.root().join("small");
    let Some(_mount) = Mount::new(&at, 32) else {
        eprintln!("skipped: this user cannot mount a tmpfs");
        return;
    };
    let out = render(&sb, &photos, &at, false);
    let stderr = text(&out.stderr);
    let line = warning(&stderr).unwrap_or_else(|| panic!("no warning: {stderr}"));
    // Said once, before the run, with the folder's full path, in the catalogue's words.
    assert!(
        line.ends_with(&format!(
            " is free in {}. It may stop part way.",
            at.display()
        )),
        "{line}"
    );
    assert!(line.contains("Warning: This export needs about "), "{line}");
    assert_eq!(stderr.matches("This export needs about").count(), 1);
    // Not a block: the run was tried, and the exact stop of H-06 ended it.
    assert_eq!(out.status.code(), Some(1), "{stderr}");
    assert!(stderr.contains("Stopped after 0 of 3."), "{stderr}");
    let at_warning = stderr.find("This export needs about").expect("warning");
    assert!(at_warning < stderr.find("Stopped").expect("stopped"));
}

#[test]
fn json_carries_a_low_space_event_and_stderr_the_sentence() {
    let sb = Sandbox::new();
    let photos = photos(&sb, 2);
    let at = sb.root().join("small-json");
    let Some(_mount) = Mount::new(&at, 32) else {
        eprintln!("skipped: this user cannot mount a tmpfs");
        return;
    };
    let out = render(&sb, &photos, &at, true);
    let records = json_lines(&out.stdout);
    let event = &records[0];
    assert_eq!(event["event"], "low_space", "{records:?}");
    assert_eq!(event["folder"], at.display().to_string());
    let needed = event["estimate_bytes"].as_u64().expect("estimate");
    let free = event["free_bytes"].as_u64().expect("free");
    assert!(free < needed * 135 / 100, "{event}");
    let message = event["message"].as_str().expect("message");
    assert_eq!(
        warning(&text(&out.stderr)).map(|l| l.ends_with(message)),
        Some(true)
    );
    assert!(message.starts_with("This export needs about "));
    // Nothing else about the warning is in the other records.
    assert!(
        records[1..]
            .iter()
            .all(|r| r.get("event").is_none_or(|e| e != "low_space"))
    );
}

/// H-08: the start of a run clears the writer's temporary files older than 24 hours from the destination
/// (`docs/26` P2-4) and leaves younger ones and other files alone.
#[test]
fn a_run_removes_stale_temporary_files_from_the_destination() {
    let sb = Sandbox::new();
    let photos = photos(&sb, 1);
    let dest = sb.root().join("stale");
    std::fs::create_dir_all(&dest).expect("dir");
    let age = |name: &str, hours: u64| {
        let path = dest.join(name);
        std::fs::write(&path, b"left by a crash").expect("write");
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .expect("open")
            .set_modified(
                std::time::SystemTime::now() - std::time::Duration::from_secs(hours * 3600),
            )
            .expect("mtime");
    };
    age(".yellowmoon-999999-0.tmp", 25);
    age(".yellowmoon-999999-1.tmp", 23);
    age("notes.txt", 500);
    let out = render(&sb, &photos, &dest, false);
    assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));
    let mut names = common::names_in(&dest);
    names.retain(|n| !n.starts_with("DSC"));
    assert_eq!(names, [".yellowmoon-999999-1.tmp", "notes.txt"]);
}
