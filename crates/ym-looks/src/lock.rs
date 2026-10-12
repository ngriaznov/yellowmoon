//! The `looks/.lock` file lock so the app and CLI can import at the same time (`docs/08-looks.md` section Library folder).
//!
//! An import reads the manifest, adds to it and writes it back. Two processes doing that at once
//! would lose one of the updates, so every writer takes an exclusive lock on `looks/.lock` first
//! and re-reads the manifest after it holds the lock. The lock is an advisory `flock` (Windows:
//! `LockFileEx`) through the `fs4` crate. It belongs to the open file, so it is released when
//! [`LibraryLock`] is dropped, and by the operating system if the process dies.
//!
//! Each [`LibraryLock::acquire`] opens its own handle, so threads of one process exclude each
//! other the same way separate processes do.

use std::fs::{self, File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

use fs4::{FileExt, TryLockError};

/// Name of the lock file inside the `looks` folder.
pub const LOCK_FILE: &str = ".lock";
/// How long [`LibraryLock::acquire`] waits by default before giving up.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(120);
/// Pause between attempts while waiting.
const POLL: Duration = Duration::from_millis(20);

/// Why the lock could not be taken.
#[derive(Debug, thiserror::Error)]
pub enum LockError {
    /// The lock file could not be created or locked.
    #[error("Could not lock the looks folder {path}: {source}")]
    Io {
        /// The lock file.
        path: PathBuf,
        /// What the system said.
        #[source]
        source: io::Error,
    },
    /// Someone else held the lock for the whole wait.
    #[error("Another import is still running. Wait for it to finish, then try again.")]
    Timeout {
        /// The lock file.
        path: PathBuf,
        /// How long we waited.
        waited: Duration,
    },
}

/// An exclusive lock on the look library. Released on drop.
#[derive(Debug)]
pub struct LibraryLock {
    file: File,
    path: PathBuf,
}

impl LibraryLock {
    fn open(looks_dir: &Path) -> Result<(File, PathBuf), LockError> {
        let path = looks_dir.join(LOCK_FILE);
        let io_error = |source| LockError::Io {
            path: path.clone(),
            source,
        };
        fs::create_dir_all(looks_dir).map_err(io_error)?;
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .map_err(io_error)?;
        Ok((file, path))
    }

    /// Takes the lock, waiting up to `timeout` for another holder to let go. Creates the `looks`
    /// folder and the lock file when they do not exist yet.
    ///
    /// # Errors
    ///
    /// [`LockError::Timeout`] when the wait runs out, [`LockError::Io`] for anything else.
    pub fn acquire(looks_dir: &Path, timeout: Duration) -> Result<Self, LockError> {
        let (file, path) = Self::open(looks_dir)?;
        let start = Instant::now();
        loop {
            match FileExt::try_lock(&file) {
                Ok(()) => return Ok(Self { file, path }),
                Err(TryLockError::WouldBlock) => {
                    let waited = start.elapsed();
                    if waited >= timeout {
                        return Err(LockError::Timeout { path, waited });
                    }
                    thread::sleep(POLL.min(timeout - waited));
                }
                Err(TryLockError::Error(source)) => return Err(LockError::Io { path, source }),
            }
        }
    }

    /// Takes the lock if it is free right now; `Ok(None)` when someone else holds it.
    ///
    /// # Errors
    ///
    /// [`LockError::Io`] when the lock file cannot be created or locked.
    pub fn try_acquire(looks_dir: &Path) -> Result<Option<Self>, LockError> {
        let (file, path) = Self::open(looks_dir)?;
        match FileExt::try_lock(&file) {
            Ok(()) => Ok(Some(Self { file, path })),
            Err(TryLockError::WouldBlock) => Ok(None),
            Err(TryLockError::Error(source)) => Err(LockError::Io { path, source }),
        }
    }

    /// The lock file's path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for LibraryLock {
    fn drop(&mut self) {
        // Closing the file releases the lock anyway; unlocking first makes the release prompt.
        let _ = FileExt::unlock(&self.file);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn second_handle_waits_for_the_first() {
        let dir = tempfile::tempdir().unwrap();
        let looks = dir.path().join("looks");
        let first = LibraryLock::acquire(&looks, Duration::from_secs(1)).unwrap();
        assert!(first.path().ends_with(LOCK_FILE));
        // A second handle in the same process must be refused while the first is alive.
        assert!(LibraryLock::try_acquire(&looks).unwrap().is_none());
        let err = LibraryLock::acquire(&looks, Duration::from_millis(60)).unwrap_err();
        assert!(matches!(err, LockError::Timeout { .. }), "{err}");
        drop(first);
        assert!(LibraryLock::try_acquire(&looks).unwrap().is_some());
    }

    #[test]
    fn waiter_gets_the_lock_when_it_is_released() {
        let dir = tempfile::tempdir().unwrap();
        let looks = dir.path().to_path_buf();
        let first = LibraryLock::acquire(&looks, Duration::from_secs(1)).unwrap();
        let (tx, rx) = std::sync::mpsc::channel();
        let waiter_dir = looks.clone();
        let waiter = thread::spawn(move || {
            let lock = LibraryLock::acquire(&waiter_dir, Duration::from_secs(60)).unwrap();
            tx.send(()).unwrap();
            drop(lock);
        });
        // Ordering, not timing: while the first lock is held the waiter cannot have it, however
        // slowly or quickly its thread starts; once it is released the waiter must get it.
        assert!(
            rx.recv_timeout(Duration::from_millis(200)).is_err(),
            "the waiter got the lock while it was held"
        );
        drop(first);
        rx.recv_timeout(Duration::from_secs(60))
            .expect("the waiter takes the lock once it is free");
        waiter.join().unwrap();
    }
}
