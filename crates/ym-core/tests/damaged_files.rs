//! The damaged-file matrix (`docs/26` P1-6, task H-03): ten valid synthetic files, one per decodable
//! variant, each damaged in nineteen ways (`ym_raw::testgen::damage`), all in one folder next to the
//! good ones.
//!
//! The session opens the folder and settles. Every damaged file is `Skipped` with a sentence of the
//! closed catalogue (`ym_raw::error`), or `Ready` when the damage only touched pixel values (the file
//! has no checksum, so nothing can tell, `docs/26` P1-5); damage that always changes the container is
//! never `Ready`. The good files are `Ready` and their export is byte for byte what the same files
//! give in a folder of their own. Nothing panics: a panic in a decode job would show as the
//! "didn't expect" sentence, which is not in the catalogue, and in the log and the panic hook.
//!
//! Choice of the matrix: all ten variants times the nineteen damages of `matrix_damages()` (190 damaged files
//! in one folder with the ten good ones, about a second), because a decoder's weak spots differ per variant and the
//! product is cheap at 96 x 64. The 1,576-file sweep of the message catalogue (the audit's 192 files,
//! grown by `docs/26` H-01) is the named case at the end, opened as one folder too.
//!
//! Permission and symlink cases are Unix-only and skip as root, which can read a mode-000 file.

mod common;

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::{Arc, Mutex, Once};
use std::time::{Duration, Instant};

use common::{captured_log, names_in, session_with_events, settle};
use ym_core::export_queue::{ExportEvent, ExportJob, ExportQueue, ExportSink, JobState};
use ym_core::pipeline::Renderer;
use ym_core::session::{ApplyScope, OpenOptions, PhotoStatus, Session};
use ym_export::ExportOptions;
use ym_raw::error::{
    MESSAGE_CUT_SHORT, MESSAGE_DAMAGED, MESSAGE_DENIED, MESSAGE_EMPTY, MESSAGE_MACOS_BLOCKED,
    MESSAGE_NOT_FOUND, MESSAGE_UNREADABLE,
};
use ym_raw::testgen::damage::{matrix_damages, sweep, valid_files, write_matrix};

/// Every sentence a skipped file may show: the catalogue of `ym_raw::error`, and the two fixed
/// sentences of a file that is not ARW and of a pixel-shift file.
fn catalogue() -> BTreeSet<&'static str> {
    BTreeSet::from([
        MESSAGE_EMPTY,
        MESSAGE_CUT_SHORT,
        MESSAGE_DAMAGED,
        MESSAGE_NOT_FOUND,
        MESSAGE_DENIED,
        MESSAGE_MACOS_BLOCKED,
        MESSAGE_UNREADABLE,
        "This isn't an ARW file. Yellowmoon opens Sony ARW files only.",
        "Pixel-shift ARQ files aren't supported. Open the individual ARW frames instead.",
    ])
}

/// Whether `reason` is one of the sentences a damaged file may give. The two `Unsupported` forms
/// (a damaged tag that reads as a mode Yellowmoon does not decode) are the open-ended part of the
/// list, as in `crates/ym-raw/tests/messages.rs`.
fn is_catalogue_sentence(reason: &str) -> bool {
    catalogue().contains(reason)
        || reason.contains("aren't supported yet")
        || reason.starts_with("Yellowmoon can't open ")
        || is_analysis_sentence(reason)
}

/// "This photo can't be analysed (the image is too small). Copy it from the card again, or skip
/// it." The estimate stage of the session (`ym_core::auto::AutoError`) refuses a file that decoded
/// to an image it cannot analyse, such as a damaged size tag that reads as a few pixels. It is not
/// in `ym_raw::error`'s catalogue and its parenthesis carries a technical detail; reported in
/// `docs/26` H-03 as built.
fn is_analysis_sentence(reason: &str) -> bool {
    reason.starts_with("This photo can't be analysed (")
        && (reason.ends_with("). Copy it from the card again, or skip it.")
            || reason.ends_with("). Skip it and open the others."))
}

/// Records every panic of the process (the decode pool catches them, so a test would not see them).
fn panics() -> &'static Mutex<Vec<String>> {
    static PANICS: Mutex<Vec<String>> = Mutex::new(Vec::new());
    static HOOK: Once = Once::new();
    HOOK.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if let Ok(mut all) = PANICS.lock() {
                all.push(info.to_string());
            }
            previous(info);
        }));
    });
    &PANICS
}

/// The photos of the session by file name, with their status.
fn statuses(session: &Session) -> BTreeMap<String, PhotoStatus> {
    session
        .state()
        .photos
        .into_iter()
        .map(|p| (p.file_name, p.status))
        .collect()
}

/// Exports every photo of `session` through the neutral renderer into `out` and returns the files
/// written, name to bytes.
fn export_all(session: &Arc<Session>, out: &Path) -> BTreeMap<String, Vec<u8>> {
    let home = tempfile::tempdir().expect("home");
    let ctx = Arc::new(
        ym_core::export::ExportContext::new(Renderer::neutral_only().expect("render thread"))
            .with_home(Some(home.path().to_path_buf())),
    );
    let (sink, rx) = ExportSink::channel();
    let queue = ExportQueue::new(ctx, sink);
    let id = queue
        .start(
            session,
            &ExportJob {
                scope: ApplyScope::All,
                options: ExportOptions {
                    destination: Some(out.to_path_buf()),
                    ..ExportOptions::default()
                },
            },
            false,
        )
        .expect("queued");
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        let left = deadline.saturating_duration_since(Instant::now());
        if rx.recv_timeout(left).expect("the queue went idle") == ExportEvent::Busy(false) {
            break;
        }
    }
    let summary = queue.summary(id).expect("summary");
    assert_eq!(summary.state, JobState::Finished);
    assert_eq!(summary.failed, 0, "{:?}", summary.errors);
    names_in(out)
        .into_iter()
        .map(|n| {
            let bytes = std::fs::read(out.join(&n)).expect("read");
            (n, bytes)
        })
        .collect()
}

#[test]
fn every_damaged_file_is_skipped_with_a_catalogue_sentence_or_ready_and_the_good_ones_are_unchanged()
 {
    let _ = panics();
    let _ = captured_log();
    let dir = tempfile::tempdir().expect("dir");
    let entries = write_matrix(dir.path()).expect("write the matrix");
    assert_eq!(entries.len(), 10 * (1 + matrix_damages().len()));

    let (session, _rx) = session_with_events();
    let session = Arc::new(session);
    let outcome = session.open(&[dir.path().to_path_buf()], OpenOptions::default());
    assert_eq!(outcome.added.len(), entries.len(), "{outcome:?}");
    settle(&session);
    let status = statuses(&session);

    let (mut skipped, mut ready, mut sentences) = (0, 0, BTreeSet::new());
    for e in &entries {
        let s = &status[&e.file];
        match (&e.damage, s) {
            (None, PhotoStatus::Ready) => {}
            (None, other) => panic!("{}: the good file is {other:?}", e.file),
            (Some(d), PhotoStatus::Skipped { reason }) => {
                assert!(
                    is_catalogue_sentence(reason),
                    "{} ({}): {reason:?} is not in the catalogue",
                    e.file,
                    d.label()
                );
                skipped += 1;
                sentences.insert(reason.clone());
            }
            (Some(d), PhotoStatus::Ready) => {
                assert!(
                    !d.always_refused(),
                    "{} ({} of {}) is Ready, but this damage always changes the container",
                    e.file,
                    d.label(),
                    e.variant
                );
                ready += 1;
            }
            (Some(_), other) => panic!("{}: {other:?} after settle", e.file),
        }
    }
    eprintln!("{skipped} skipped, {ready} ready, sentences: {sentences:#?}");
    // The matrix reaches the three damage sentences and the foreign-file one, and damage that only
    // touches pixel values is shown (not refused).
    for sentence in [MESSAGE_EMPTY, MESSAGE_CUT_SHORT, MESSAGE_DAMAGED] {
        assert!(sentences.contains(sentence), "never produced {sentence:?}");
    }
    assert!(sentences.contains("This isn't an ARW file. Yellowmoon opens Sony ARW files only."));
    assert!(ready > 0, "some damage lies in pixel values only");

    // No panic anywhere: not in the hook, not in the log, and no photo says "didn't expect".
    assert!(panics().lock().expect("panics").is_empty());
    let log = captured_log();
    assert!(!log.contains("panicked"), "{log}");
    assert!(!log.contains("a decode job panicked"));

    // The good files give the same bytes as in a folder of their own, and as each one alone.
    let out = tempfile::tempdir().expect("out");
    let together = export_all(&session, out.path());
    let good_names: Vec<&str> = entries
        .iter()
        .filter(|e| e.damage.is_none())
        .map(|e| e.file.as_str())
        .collect();
    let only_good = tempfile::tempdir().expect("good folder");
    let valid = valid_files();
    for (variant, bytes) in &valid {
        std::fs::write(only_good.path().join(format!("good_{variant}.ARW")), bytes).expect("write");
    }
    let (clean, _rx) = session_with_events();
    let clean = Arc::new(clean);
    clean.open(&[only_good.path().to_path_buf()], OpenOptions::default());
    settle(&clean);
    let clean_out = tempfile::tempdir().expect("out");
    let alone = export_all(&clean, clean_out.path());
    assert_eq!(alone.len(), good_names.len());
    for name in &good_names {
        let stem = name.trim_end_matches(".ARW");
        let output = format!("{stem}_neutral.jpg");
        assert!(
            alone.get(&output).is_some_and(|b| !b.is_empty()),
            "{output} was not written alone"
        );
        assert_eq!(
            together.get(&output),
            alone.get(&output),
            "{name} changed next to damaged files"
        );
    }
}

/// The audit's sweep as a named case: the foreign files and 157 damages of each variant (1,576 files;
/// the audit's 192 grew with `docs/26` H-01) in one folder with the ten good files. Every file ends
/// `Skipped` with a catalogue sentence or `Ready`; the good files are `Ready`; nothing panics.
#[test]
fn the_audit_sweep_of_1576_files_opens_in_one_folder_without_a_panic_or_a_foreign_sentence() {
    let _ = panics();
    let _ = captured_log();
    let dir = tempfile::tempdir().expect("dir");
    let files = sweep();
    assert_eq!(files.len(), 1576);
    for (i, (_, bytes)) in files.iter().enumerate() {
        std::fs::write(dir.path().join(format!("sweep_{i:04}.ARW")), bytes).expect("write");
    }
    for (variant, bytes) in valid_files() {
        std::fs::write(dir.path().join(format!("good_{variant}.ARW")), bytes).expect("write");
    }
    let (session, _rx) = session_with_events();
    let outcome = session.open(&[dir.path().to_path_buf()], OpenOptions::default());
    assert_eq!(outcome.added.len(), 1586, "{outcome:?}");
    settle(&session);
    let (mut ready, mut skipped, mut analysis) = (0, 0, BTreeSet::<String>::new());
    for (name, status) in statuses(&session) {
        match status {
            PhotoStatus::Ready => ready += 1,
            PhotoStatus::Skipped { reason } => {
                assert!(!name.starts_with("good_"), "{name} is skipped: {reason:?}");
                if is_analysis_sentence(&reason) {
                    analysis.insert(reason.clone());
                }
                assert!(
                    is_catalogue_sentence(&reason),
                    "{name}: {reason:?} is not in the catalogue"
                );
                skipped += 1;
            }
            other => panic!("{name}: {other:?} after settle"),
        }
    }
    eprintln!("sweep: {ready} ready, {skipped} skipped, analysis sentences {analysis:#?}");
    assert!(ready >= 10 && skipped > 0);
    assert!(panics().lock().expect("panics").is_empty());
    assert!(!captured_log().contains("a decode job panicked"));
}

/// A file without read permission is skipped with the "isn't allowed" sentence and the others open.
/// Skipped when the user can read a mode-000 file (root), as in `session.rs` for a folder.
#[cfg(unix)]
#[test]
fn a_file_without_read_permission_is_skipped_and_the_others_open() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().expect("dir");
    let valid = valid_files().swap_remove(0).1;
    std::fs::write(dir.path().join("a_good.ARW"), &valid).expect("write");
    let locked = dir.path().join("b_locked.ARW");
    std::fs::write(&locked, &valid).expect("write");
    std::fs::write(dir.path().join("c_good.ARW"), &valid).expect("write");
    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).expect("chmod");
    let readable = std::fs::File::open(&locked).is_ok();
    let (session, _rx) = session_with_events();
    let outcome = session.open(&[dir.path().to_path_buf()], OpenOptions::default());
    settle(&session);
    let status = statuses(&session);
    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o644)).expect("chmod");
    if readable {
        eprintln!("skipped: this user can read a file with mode 000 (root)");
        return;
    }
    assert_eq!(outcome.added.len(), 3, "{outcome:?}");
    assert_eq!(status["a_good.ARW"], PhotoStatus::Ready);
    assert_eq!(status["c_good.ARW"], PhotoStatus::Ready);
    match &status["b_locked.ARW"] {
        PhotoStatus::Skipped { reason } => assert!(
            reason == MESSAGE_DENIED || reason == MESSAGE_MACOS_BLOCKED,
            "{reason:?}"
        ),
        other => panic!("b_locked.ARW: {other:?}"),
    }
}

/// Symlinks in a folder: one to a good file opens like the file; a dangling one, a loop and one to a
/// folder never stop the others, and none panics. Runs as root too.
#[cfg(unix)]
#[test]
fn symlinks_in_a_folder_open_or_are_skipped_and_never_stop_the_others() {
    use std::os::unix::fs::symlink;
    let _ = panics();
    let dir = tempfile::tempdir().expect("dir");
    let valid = valid_files().swap_remove(0).1;
    std::fs::write(dir.path().join("a_good.ARW"), &valid).expect("write");
    std::fs::write(dir.path().join("target.dat"), &valid).expect("write");
    symlink("target.dat", dir.path().join("b_link.ARW")).expect("link");
    symlink("nowhere.dat", dir.path().join("c_dangling.ARW")).expect("link");
    symlink("d_loop.ARW", dir.path().join("d_loop.ARW")).expect("link");
    std::fs::create_dir(dir.path().join("sub")).expect("dir");
    symlink("sub", dir.path().join("e_folder.ARW")).expect("link");
    std::fs::write(dir.path().join("z_good.ARW"), &valid).expect("write");
    let (session, _rx) = session_with_events();
    let outcome = session.open(&[dir.path().to_path_buf()], OpenOptions::default());
    settle(&session);
    let status = statuses(&session);
    // The folder walk adds the good files and the link to one; the dangling link, the loop, the link
    // to a folder and `target.dat` are not ARW photos and are ignored, as other non-files are.
    assert_eq!(outcome.added.len(), 3, "{outcome:?}");
    assert_eq!(outcome.ignored, 4, "{outcome:?}");
    assert!(outcome.missing.is_empty() && outcome.unreadable.is_empty());
    assert_eq!(
        status.keys().map(String::as_str).collect::<Vec<_>>(),
        ["a_good.ARW", "b_link.ARW", "z_good.ARW"]
    );
    assert!(status.values().all(|s| *s == PhotoStatus::Ready));
    // Named one by one, the broken ones are reported, not opened and not a panic.
    let named: Vec<_> = ["c_dangling.ARW", "d_loop.ARW", "e_folder.ARW"]
        .iter()
        .map(|n| dir.path().join(n))
        .collect();
    let again = session.open(&named, OpenOptions::default());
    // (The link to a folder is a folder: it is walked, and this one is empty.)
    assert!(again.added.is_empty(), "{again:?}");
    assert_eq!(again.missing, [named[0].clone(), named[1].clone()]);
    assert!(panics().lock().expect("panics").is_empty());
}
