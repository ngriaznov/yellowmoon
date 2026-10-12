//! The application log: `logs/yellowmoon.log` in the app folder, local only, rotated at 5 MB with
//! three files kept (`docs/01` §Persistence, §Errors and logging).
//!
//! Info level is user actions and job summaries; debug adds stage timings (`YELLOWMOON_LOG=debug`).
//! Nothing is sent anywhere, and file paths are not logged at info level (a bug report may carry the
//! log, `docs/01`). Development builds also print to stderr.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};

use tracing_subscriber::EnvFilter;
use tracing_subscriber::fmt::MakeWriter;

/// The size at which the log is rotated.
pub const MAX_BYTES: u64 = 5 * 1024 * 1024;

/// How many files are kept: `yellowmoon.log`, `yellowmoon.log.1`, `yellowmoon.log.2`.
pub const KEPT: usize = 3;

/// A log file that starts a new file when it would grow past a limit.
#[derive(Debug)]
pub struct RotatingFile {
    path: PathBuf,
    max_bytes: u64,
    kept: usize,
    file: Option<File>,
    size: u64,
}

impl RotatingFile {
    /// A log at `path` that keeps `kept` files of at most `max_bytes` each. The file is opened (and
    /// its folder made) when the first line is written.
    #[must_use]
    pub fn new(path: PathBuf, max_bytes: u64, kept: usize) -> Self {
        Self {
            path,
            max_bytes,
            kept: kept.max(1),
            file: None,
            size: 0,
        }
    }

    fn numbered(&self, n: usize) -> PathBuf {
        let mut name = self.path.as_os_str().to_owned();
        name.push(format!(".{n}"));
        PathBuf::from(name)
    }

    fn open(&mut self) -> io::Result<()> {
        if let Some(dir) = self.path.parent() {
            fs::create_dir_all(dir)?;
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        self.size = file.metadata().map_or(0, |m| m.len());
        self.file = Some(file);
        Ok(())
    }

    /// `log.1` becomes `log.2`, `log` becomes `log.1`, the oldest goes.
    fn rotate(&mut self) -> io::Result<()> {
        self.file = None;
        for n in (1..self.kept).rev() {
            let from = if n == 1 {
                self.path.clone()
            } else {
                self.numbered(n - 1)
            };
            if from.exists() {
                let to = self.numbered(n);
                let _ = fs::remove_file(&to);
                fs::rename(&from, &to)?;
            }
        }
        if self.kept == 1 {
            let _ = fs::remove_file(&self.path);
        }
        self.open()
    }
}

impl Write for RotatingFile {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if self.file.is_none() {
            self.open()?;
        }
        if self.size > 0 && self.size + buf.len() as u64 > self.max_bytes {
            self.rotate()?;
        }
        let file = self
            .file
            .as_mut()
            .ok_or_else(|| io::Error::other("log closed"))?;
        file.write_all(buf)?;
        self.size += buf.len() as u64;
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        match &mut self.file {
            Some(f) => f.flush(),
            None => Ok(()),
        }
    }
}

/// Hands the shared log file to the subscriber, and echoes to stderr in development builds.
#[derive(Clone)]
struct Shared {
    file: Arc<Mutex<RotatingFile>>,
    stderr: bool,
}

struct Line<'a> {
    shared: &'a Shared,
}

impl Write for Line<'_> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if self.shared.stderr {
            let _ = io::stderr().write_all(buf);
        }
        self.shared
            .file
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.shared
            .file
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .flush()
    }
}

impl<'a> MakeWriter<'a> for Shared {
    type Writer = Line<'a>;

    fn make_writer(&'a self) -> Self::Writer {
        Line { shared: self }
    }
}

/// Starts logging to `<app_root>/logs/yellowmoon.log`. Does nothing if a subscriber is already set
/// (tests). The folder is made on the first line, not here.
pub fn init(app_root: &Path) {
    let filter =
        EnvFilter::try_from_env("YELLOWMOON_LOG").unwrap_or_else(|_| EnvFilter::new("info"));
    let shared = Shared {
        file: Arc::new(Mutex::new(RotatingFile::new(
            app_root.join("logs").join("yellowmoon.log"),
            MAX_BYTES,
            KEPT,
        ))),
        stderr: cfg!(debug_assertions),
    };
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_ansi(false)
        .with_writer(shared)
        .try_init();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_log_rotates_and_keeps_three_files() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("logs").join("yellowmoon.log");
        let mut log = RotatingFile::new(path.clone(), 100, 3);
        // Nothing exists until the first line.
        assert!(!path.exists());
        for round in 0..6 {
            for _ in 0..4 {
                writeln!(log, "line {round:02} {}", "x".repeat(20)).expect("write");
            }
        }
        let names: Vec<String> = fs::read_dir(path.parent().expect("dir"))
            .expect("read")
            .map(|e| e.expect("entry").file_name().to_string_lossy().into_owned())
            .collect();
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(
            sorted,
            ["yellowmoon.log", "yellowmoon.log.1", "yellowmoon.log.2"],
            "{names:?}"
        );
        for n in &sorted {
            let len = fs::metadata(path.parent().expect("dir").join(n))
                .expect("meta")
                .len();
            assert!(len <= 100, "{n} is {len} bytes");
        }
        // The newest lines are in the current file, older ones in the numbered files.
        let current = fs::read_to_string(&path).expect("current");
        assert!(current.contains("line 05"));
        let oldest = fs::read_to_string(path.with_extension("log.2")).expect("oldest");
        assert!(oldest.contains("line "));
        assert!(!oldest.contains("line 05"));
    }

    #[test]
    fn a_single_kept_file_is_simply_restarted() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("a.log");
        let mut log = RotatingFile::new(path.clone(), 30, 1);
        for i in 0..10 {
            writeln!(log, "number {i}").expect("write");
        }
        assert!(fs::metadata(&path).expect("meta").len() <= 30);
        assert!(!dir.path().join("a.log.1").exists());
    }
}
