//! `ym render` stops at a failure of the destination (`docs/26` P2-2, task H-06): one error line, then
//! "Stopped after k of n. m not exported.", exit code 1, and with `--json` a `stopped` event after the
//! photo's own record. The photos after the one that hit the full disk are not rendered.
//!
//! A real `ENOSPC` needs a small `tmpfs`, which needs `mount` (Linux, usually root); the tests say so and
//! return when it is refused. `ym-core/tests/export_stop.rs` drives the same stop with a write fault on
//! any machine.

mod common;

use std::path::{Path, PathBuf};
use std::process::Command;

use common::{Sandbox, json_records, lines, names_in, text};

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

const PHOTOS: usize = 5;

fn photos(sb: &Sandbox) -> Vec<PathBuf> {
    (0..PHOTOS)
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

/// The size of the first photo's file, to size a disk that holds it and not the second.
fn size_of_one(sb: &Sandbox, photos: &[PathBuf]) -> u64 {
    let dir = sb.root().join("size-probe");
    std::fs::create_dir_all(&dir).expect("dir");
    let out = render(sb, &photos[..1], &dir, false);
    assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));
    let name = names_in(&dir).pop().expect("the probe file");
    let size = std::fs::metadata(dir.join(name)).expect("size").len();
    std::fs::remove_dir_all(&dir).expect("clean");
    size
}

#[test]
fn a_full_disk_stops_the_run_with_two_lines_and_exit_1() {
    let sb = Sandbox::new();
    let photos = photos(&sb);
    let at = sb.root().join("full");
    // Too small for any of the files.
    let Some(_mount) = Mount::new(&at, 32) else {
        eprintln!("skipped: this user cannot mount a tmpfs");
        return;
    };
    let out = render(&sb, &photos, &at, false);
    assert_eq!(out.status.code(), Some(1));
    assert!(
        out.stdout.is_empty(),
        "no file was written: {}",
        text(&out.stdout)
    );
    let stderr = text(&out.stderr);
    let lines = lines(stderr.as_bytes());
    let errors: Vec<_> = lines.iter().filter(|l| l.contains("is full")).collect();
    assert_eq!(
        errors.len(),
        1,
        "one error line, not one per photo: {stderr}"
    );
    assert!(
        errors[0].ends_with(&format!(
            "Stopped: the disk holding {} is full. Free some space, then run the command again with --skip-existing.",
            at.display()
        )),
        "{stderr}"
    );
    assert!(
        lines
            .iter()
            .any(|l| l == "Stopped after 0 of 5. 5 not exported."),
        "{stderr}"
    );
    assert!(
        names_in(&at).is_empty(),
        "no temporary file is left: {:?}",
        names_in(&at)
    );
}

#[test]
fn json_says_stopped_after_the_record_of_the_photo_it_hit() {
    let sb = Sandbox::new();
    let photos = photos(&sb);
    let one = size_of_one(&sb, &photos);
    let at = sb.root().join("one-fits");
    // Room for the first file and not for the second (`size` is in KiB; a tmpfs counts pages).
    let Some(_mount) = Mount::new(&at, (one * 3 / 2).div_ceil(1024)) else {
        eprintln!("skipped: this user cannot mount a tmpfs");
        return;
    };
    let out = render(&sb, &photos, &at, true);
    assert_eq!(out.status.code(), Some(1));
    let records = json_records(&out.stdout);
    assert_eq!(records.len(), 3, "{records:?}");
    assert_eq!(records[0]["status"], "ok");
    assert_eq!(records[1]["status"], "error");
    assert_eq!(records[1]["error"]["kind"], "disk_full");
    assert_eq!(
        records[2],
        serde_json::json!({"event": "stopped", "kind": "disk_full", "exported": 1, "not_tried": 3}),
    );
    // The first file stays; nothing else, no temporary file.
    let kept = names_in(&at);
    assert_eq!(kept.len(), 1, "{kept:?}");
    assert!(
        kept[0].starts_with("DSC00001_") && kept[0].ends_with(".png"),
        "{kept:?}"
    );
    let stderr = text(&out.stderr);
    assert!(
        lines(stderr.as_bytes())
            .iter()
            .all(|l| !l.contains("Stopped after") || *l == "Stopped after 1 of 5. 4 not exported."),
        "{stderr}"
    );
}

/// A folder that is not there ends the run at the first photo, before it is decoded.
#[test]
fn a_folder_that_is_gone_stops_the_run_too() {
    if !cfg!(target_os = "linux") {
        eprintln!("skipped: needs /proc");
        return;
    }
    let sb = Sandbox::new();
    let photos = photos(&sb);
    let gone = Path::new("/proc/ym-no-such-folder/exports");
    if std::fs::create_dir_all(gone).is_ok() {
        eprintln!("skipped: /proc accepts new folders here");
        return;
    }
    let out = render(&sb, &photos, gone, true);
    assert_eq!(out.status.code(), Some(1));
    let records = json_records(&out.stdout);
    assert_eq!(records.len(), 2, "{records:?}");
    assert_eq!(records[0]["error"]["kind"], "destination");
    assert_eq!(
        records[1],
        serde_json::json!({"event": "stopped", "kind": "destination", "exported": 0, "not_tried": 4}),
    );
    let stderr = text(&out.stderr);
    assert!(
        stderr.is_empty() || !stderr.contains("5 failed"),
        "{stderr}"
    );
}
