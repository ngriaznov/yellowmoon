//! The Downloads watcher against a real folder, with short timings. The stability logic itself is
//! unit-tested with made-up clocks in `src/import/watch.rs`.

mod support;

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use support::small_db;
use ym_looks::import::watch::{DownloadsWatcher, FoundZip, WatchConfig, ZipNameMatcher};

fn fast() -> WatchConfig {
    WatchConfig {
        poll_interval: Duration::from_millis(20),
        stable_for: Duration::from_millis(700),
        report_existing: false,
    }
}

fn start(
    folder: &std::path::Path,
    config: &WatchConfig,
) -> (DownloadsWatcher, mpsc::Receiver<(FoundZip, Instant)>) {
    let (tx, rx) = mpsc::channel();
    let matcher = ZipNameMatcher::new(&small_db(1), None);
    let watcher = DownloadsWatcher::start(folder, matcher, config, move |found| {
        let _ = tx.send((found, Instant::now()));
    })
    .unwrap();
    (watcher, rx)
}

#[test]
fn a_download_is_reported_once_after_its_size_stops_changing() {
    let dir = tempfile::tempdir().unwrap();
    let (watcher, rx) = start(dir.path(), &fast());

    // A browser writing the zip in pieces; other files that must not be reported.
    fs::write(dir.path().join("holiday.zip"), b"not an ARRI package").unwrap();
    fs::write(
        dir.path().join("ARRI_Look_Library_LogC4.zip.crdownload"),
        b"partial",
    )
    .unwrap();
    fs::write(dir.path().join("notes.txt"), b"hello").unwrap();
    let target = dir.path().join("ARRI_Look_Library_LogC4.zip");
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&target)
        .unwrap();
    let mut last_write = Instant::now();
    for chunk in 0..8_u8 {
        file.write_all(&[chunk; 4096]).unwrap();
        file.flush().unwrap();
        last_write = Instant::now();
        thread::sleep(Duration::from_millis(60));
    }
    drop(file);
    let (found, at) = rx
        .recv_timeout(Duration::from_secs(10))
        .expect("the zip is reported");
    assert_eq!(found.path, target);
    assert_eq!(found.bytes, 8 * 4096);
    assert_eq!(found.packages, vec!["look-library-logc4"]);
    // Not before it had been quiet for the stable period (a little slack for timer rounding).
    assert!(
        at.duration_since(last_write) >= Duration::from_millis(650),
        "reported only {:?} after the last write",
        at.duration_since(last_write)
    );
    // Exactly once, and nothing else.
    assert!(
        rx.recv_timeout(Duration::from_millis(1500)).is_err(),
        "reported again"
    );
    watcher.stop();
}

#[test]
fn a_zip_that_was_already_there_is_ignored_unless_asked_for() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(
        dir.path().join("ARRI_Display_LUTs_LogC4.zip"),
        b"old download",
    )
    .unwrap();
    let (quiet, rx) = start(dir.path(), &fast());
    assert!(rx.recv_timeout(Duration::from_millis(1800)).is_err());
    quiet.stop();

    let eager = WatchConfig {
        report_existing: true,
        ..fast()
    };
    let (watcher, rx) = start(dir.path(), &eager);
    let (found, _) = rx.recv_timeout(Duration::from_secs(10)).expect("reported");
    assert_eq!(found.packages, vec!["display-luts-logc4"]);
    drop(watcher);
}

#[test]
fn a_new_download_under_the_same_name_is_reported_again() {
    let dir = tempfile::tempdir().unwrap();
    let (watcher, rx) = start(dir.path(), &fast());
    let target = dir.path().join("ARRI_Look_Library_LogC3.zip");
    fs::write(&target, vec![1_u8; 1000]).unwrap();
    rx.recv_timeout(Duration::from_secs(10)).expect("first");
    fs::write(&target, vec![2_u8; 2500]).unwrap();
    let (again, _) = rx.recv_timeout(Duration::from_secs(10)).expect("second");
    assert_eq!(again.bytes, 2500);
    watcher.stop();
}

#[test]
fn only_the_chosen_packages_are_watched_for() {
    let dir = tempfile::tempdir().unwrap();
    let (tx, rx) = mpsc::channel();
    let only = ["display-luts-logc4".to_string()];
    let matcher = ZipNameMatcher::new(&small_db(1), Some(&only));
    let watcher = DownloadsWatcher::start(dir.path(), matcher, &fast(), move |f| {
        let _ = tx.send(f);
    })
    .unwrap();
    fs::write(dir.path().join("ARRI_Look_Library_LogC4.zip"), b"looks").unwrap();
    fs::write(dir.path().join("ARRI_Display_LUTs_LogC4.zip"), b"displays").unwrap();
    let found = rx
        .recv_timeout(Duration::from_secs(10))
        .expect("display zip");
    assert!(found.path.ends_with("ARRI_Display_LUTs_LogC4.zip"));
    assert!(rx.recv_timeout(Duration::from_millis(1500)).is_err());
    watcher.stop();
}

#[test]
fn an_unreadable_folder_is_reported_when_watching_starts() {
    let dir = tempfile::tempdir().unwrap();
    let matcher = ZipNameMatcher::new(&small_db(1), None);
    let err = DownloadsWatcher::start(dir.path().join("Downloads"), matcher, &fast(), |_| {})
        .unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
}

#[test]
fn stopping_is_prompt_and_dropping_stops_too() {
    let dir = tempfile::tempdir().unwrap();
    let slow = WatchConfig {
        poll_interval: Duration::from_secs(30),
        ..fast()
    };
    let (watcher, _rx) = start(dir.path(), &slow);
    let begun = Instant::now();
    watcher.stop();
    assert!(
        begun.elapsed() < Duration::from_secs(2),
        "{:?}",
        begun.elapsed()
    );
    let (watcher, _rx) = start(dir.path(), &slow);
    let begun = Instant::now();
    drop(watcher);
    assert!(begun.elapsed() < Duration::from_secs(2));
}
