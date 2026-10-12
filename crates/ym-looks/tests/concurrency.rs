//! Concurrent imports: the app and the CLI may import at the same time (`docs/08-looks.md`,
//! "Writes are atomic ... and take `looks/.lock`"). Threads with separate library handles and
//! separate processes both have to end with every import in the manifest.

mod support;

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Duration;

use support::{cube_text, no_progress, open_library, small_db, tree};
use ym_color::look::LookInput;
use ym_looks::import::{ImportError, ImportOptions, import_path};
use ym_looks::library::{Library, LookId};
use ym_looks::lock::{LibraryLock, LockError};

fn answer() -> ImportOptions {
    ImportOptions {
        input: Some(LookInput::LogC4),
        imported_at: Some("2026-11-02T10:14:00Z".to_string()),
        ..ImportOptions::default()
    }
}

fn write_cube(dir: &Path, name: &str, seed: u32) -> PathBuf {
    let path = dir.join(format!("{name}.cube"));
    fs::write(&path, cube_text(5, seed, name)).unwrap();
    path
}

fn user_ids(root: &Path) -> Vec<String> {
    open_library(root)
        .looks()
        .iter()
        .filter(|l| l.id.prefix() == "user")
        .map(|l| l.id.to_string())
        .collect()
}

#[test]
fn threads_with_separate_handles_lose_no_updates() {
    const THREADS: usize = 8;
    const PER_THREAD: usize = 4;
    for round in 0..3 {
        let work = tempfile::tempdir().unwrap();
        let root = work.path().join("app");
        let barrier = Arc::new(Barrier::new(THREADS));
        let handles: Vec<_> = (0..THREADS)
            .map(|t| {
                let (root, barrier, src) = (
                    root.clone(),
                    Arc::clone(&barrier),
                    work.path().to_path_buf(),
                );
                thread::spawn(move || {
                    // Each thread has its own Library and so its own lock handle.
                    let mut library = open_library(&root);
                    let db = small_db(1);
                    barrier.wait();
                    for i in 0..PER_THREAD {
                        let name = format!("t{t}-look{i}");
                        let path = write_cube(&src, &name, (t * 10 + i) as u32 + 1);
                        import_path(&mut library, &db, &path, &answer(), &mut no_progress())
                            .unwrap_or_else(|e| panic!("{name}: {e}"));
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        let ids = user_ids(&root);
        assert_eq!(
            ids.len(),
            THREADS * PER_THREAD,
            "round {round}: lost updates: {ids:?}"
        );
        let mut library = open_library(&root);
        assert!(library.verify(&mut no_progress()).unwrap().is_clean());
        assert!(
            tree(&root).iter().all(|f| !f.contains(".tmp-")),
            "{:?}",
            tree(&root)
        );
        assert_eq!(
            tree(&root).iter().filter(|f| f.ends_with(".cube")).count(),
            THREADS * PER_THREAD
        );
    }
}

#[test]
fn a_stale_library_handle_does_not_overwrite_a_newer_manifest() {
    // The scenario behind the lock: the app opened the library long ago; the CLI imports; then
    // the app imports. The app's in-memory manifest is stale, but it re-reads under the lock.
    let work = tempfile::tempdir().unwrap();
    let root = work.path().join("app");
    let db = small_db(1);
    let mut app = open_library(&root);
    let mut cli = open_library(&root);
    let first = write_cube(work.path(), "from-cli", 1);
    let second = write_cube(work.path(), "from-app", 2);
    import_path(&mut cli, &db, &first, &answer(), &mut no_progress()).unwrap();
    assert!(
        app.get(&LookId::user("from-cli")).is_none(),
        "the app has not noticed yet"
    );
    import_path(&mut app, &db, &second, &answer(), &mut no_progress()).unwrap();
    assert_eq!(user_ids(&root), vec!["user:from-app", "user:from-cli"]);
    assert!(
        app.get(&LookId::user("from-cli")).is_some(),
        "and now it has"
    );
}

/// Entry point of the child processes spawned by `separate_processes_lose_no_updates`. A plain
/// run of the test suite has none of the environment variables, so it does nothing.
#[test]
fn child_import() {
    let (Ok(root), Ok(file)) = (
        env::var("YM_LOOKS_CHILD_ROOT"),
        env::var("YM_LOOKS_CHILD_FILE"),
    ) else {
        return;
    };
    let mut library = Library::open(&root, None).unwrap();
    let db = small_db(1);
    import_path(
        &mut library,
        &db,
        Path::new(&file),
        &answer(),
        &mut no_progress(),
    )
    .unwrap_or_else(|e| panic!("child import of {file} failed: {e}"));
}

#[test]
fn separate_processes_lose_no_updates() {
    const CHILDREN: usize = 8;
    let work = tempfile::tempdir().unwrap();
    let root = work.path().join("app");
    let exe = env::current_exe().unwrap();
    let mut running = Vec::new();
    for i in 0..CHILDREN {
        let file = write_cube(work.path(), &format!("proc{i}"), i as u32 + 1);
        let child = Command::new(&exe)
            .args(["child_import", "--exact", "--nocapture", "--test-threads=1"])
            .env("YM_LOOKS_CHILD_ROOT", &root)
            .env("YM_LOOKS_CHILD_FILE", &file)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        running.push(child);
    }
    for child in running {
        let out = child.wait_with_output().unwrap();
        assert!(
            out.status.success(),
            "child failed:\n{}\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        // The child really ran the import (it printed the test as run, not filtered out).
        assert!(String::from_utf8_lossy(&out.stdout).contains("1 passed"));
    }
    let ids = user_ids(&root);
    assert_eq!(
        ids.len(),
        CHILDREN,
        "lost updates across processes: {ids:?}"
    );
    let mut library = open_library(&root);
    assert!(library.verify(&mut no_progress()).unwrap().is_clean());
}

#[test]
fn a_second_importer_waits_for_the_lock_and_then_succeeds() {
    let work = tempfile::tempdir().unwrap();
    let root = work.path().join("app");
    let looks = root.join("looks");
    let path = write_cube(work.path(), "late", 1);
    let held = LibraryLock::acquire(&looks, Duration::from_secs(1)).unwrap();

    // With a short patience it gives up with a plain message ...
    let mut library = open_library(&root);
    let impatient = ImportOptions {
        lock_timeout: Duration::from_millis(80),
        ..answer()
    };
    let err = import_path(
        &mut library,
        &small_db(1),
        &path,
        &impatient,
        &mut no_progress(),
    )
    .unwrap_err();
    assert!(
        matches!(err, ImportError::Lock(LockError::Timeout { .. })),
        "{err}"
    );
    assert!(err.to_string().contains("Another import is still running"));
    assert!(user_ids(&root).is_empty());

    // ... and with more it waits its turn.
    let (root2, path2) = (root.clone(), path.clone());
    let waiter = thread::spawn(move || {
        let mut library = open_library(&root2);
        let patient = ImportOptions {
            lock_timeout: Duration::from_secs(20),
            ..answer()
        };
        import_path(
            &mut library,
            &small_db(1),
            &path2,
            &patient,
            &mut no_progress(),
        )
        .unwrap();
    });
    thread::sleep(Duration::from_millis(200));
    assert!(
        user_ids(&root).is_empty(),
        "nothing is written while the lock is held"
    );
    drop(held);
    waiter.join().unwrap();
    assert_eq!(user_ids(&root), vec!["user:late"]);
}

#[test]
fn removing_takes_the_same_lock() {
    let work = tempfile::tempdir().unwrap();
    let root = work.path().join("app");
    let mut library = open_library(&root);
    let path = write_cube(work.path(), "gone", 1);
    import_path(
        &mut library,
        &small_db(1),
        &path,
        &answer(),
        &mut no_progress(),
    )
    .unwrap();
    // Remove concurrently with imports of other files from other threads.
    let barrier = Arc::new(Barrier::new(5));
    let importers: Vec<_> = (0..4)
        .map(|t| {
            let (root, barrier, src) = (
                root.clone(),
                Arc::clone(&barrier),
                work.path().to_path_buf(),
            );
            thread::spawn(move || {
                let mut lib = open_library(&root);
                let p = write_cube(&src, &format!("new{t}"), t as u32 + 10);
                barrier.wait();
                import_path(&mut lib, &small_db(1), &p, &answer(), &mut no_progress()).unwrap();
            })
        })
        .collect();
    barrier.wait();
    library.remove(&LookId::user("gone")).unwrap();
    for h in importers {
        h.join().unwrap();
    }
    let ids = user_ids(&root);
    assert_eq!(
        ids,
        vec!["user:new0", "user:new1", "user:new2", "user:new3"]
    );
}
