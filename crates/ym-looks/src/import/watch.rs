//! The wizard's "Watch Downloads" switch: report new `.zip` files that look like ARRI packages once they have finished downloading (`docs/08-looks.md` section Wizard, step 2).
//!
//! A background thread lists the folder every [`WatchConfig::poll_interval`] and keeps, for each
//! `.zip` whose name matches a package pattern, the size and modification time it last saw. A
//! file is reported once, after both have stayed the same for [`WatchConfig::stable_for`] (two
//! seconds by default, which covers browsers that rename or pause at the end of a download). A
//! file is reported again only if it changes afterwards (a new download under the same name).
//!
//! Polling instead of FSEvents/`notify`: the folder is small, the check is one directory listing
//! per half second while the wizard window is open, it behaves the same on every platform and
//! under every browser's way of writing a download, and it needs no dependency. (`notify` is
//! CC0-1.0, which `deny.toml` allows, so it could replace this later without a licence question.)
//!
//! The first listing is a baseline: zips already in the folder are not reported unless
//! [`WatchConfig::report_existing`] is set, so the wizard does not offer last year's download.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime};

use crate::recognize::PackageDb;

/// How often the folder is listed by default.
pub const DEFAULT_POLL_INTERVAL: Duration = Duration::from_millis(500);
/// How long a file's size must stay the same before it is reported, by default.
pub const DEFAULT_STABLE_FOR: Duration = Duration::from_secs(2);
/// The longest the watcher thread sleeps between checks of its stop flag.
const SLEEP_SLICE: Duration = Duration::from_millis(25);

/// Settings for [`DownloadsWatcher`].
#[derive(Debug, Clone)]
pub struct WatchConfig {
    /// Time between folder listings.
    pub poll_interval: Duration,
    /// How long size and modification time must stay unchanged.
    pub stable_for: Duration,
    /// Also report zips that were already in the folder when watching started.
    pub report_existing: bool,
}

impl Default for WatchConfig {
    fn default() -> Self {
        Self {
            poll_interval: DEFAULT_POLL_INTERVAL,
            stable_for: DEFAULT_STABLE_FOR,
            report_existing: false,
        }
    }
}

/// A finished download that looks like one of the packages being waited for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FoundZip {
    /// The zip.
    pub path: PathBuf,
    /// Its size in bytes.
    pub bytes: u64,
    /// The ids of the packages its name matches.
    pub packages: Vec<String>,
}

/// Matches zip file names against package patterns.
#[derive(Debug, Clone)]
pub struct ZipNameMatcher {
    db: PackageDb,
    only: Option<Vec<String>>,
}

impl ZipNameMatcher {
    /// Matches against every package in `db`, or only those in `only` (the ids the user chose
    /// in step 1 of the wizard).
    #[must_use]
    pub fn new(db: &PackageDb, only: Option<&[String]>) -> Self {
        Self {
            db: db.clone(),
            only: only.map(<[String]>::to_vec),
        }
    }

    /// The ids of the packages `file_name` could be, empty if it is not a `.zip` or matches none.
    #[must_use]
    pub fn packages_for(&self, file_name: &str) -> Vec<String> {
        let is_zip = Path::new(file_name)
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("zip"));
        if !is_zip {
            return Vec::new();
        }
        self.db
            .packages_for_zip_name(file_name)
            .into_iter()
            .filter(|p| self.only.as_ref().is_none_or(|ids| ids.contains(&p.id)))
            .map(|p| p.id.clone())
            .collect()
    }
}

/// What the watcher saw of one file in one listing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Observation {
    /// The file.
    pub path: PathBuf,
    /// Its size in bytes.
    pub size: u64,
    /// Its modification time, when the platform says.
    pub modified: Option<SystemTime>,
}

impl Observation {
    fn stamp(&self) -> (u64, Option<SystemTime>) {
        (self.size, self.modified)
    }
}

#[derive(Debug)]
struct Tracked {
    stamp: (u64, Option<SystemTime>),
    since: Instant,
}

/// The "has it stopped growing?" logic, separate from the thread and the clock so it can be tested
/// with made-up times.
#[derive(Debug)]
pub struct StabilityTracker {
    stable_for: Duration,
    tracking: HashMap<PathBuf, Tracked>,
    reported: HashMap<PathBuf, (u64, Option<SystemTime>)>,
}

impl StabilityTracker {
    /// A tracker that reports files unchanged for `stable_for`.
    #[must_use]
    pub fn new(stable_for: Duration) -> Self {
        Self {
            stable_for,
            tracking: HashMap::new(),
            reported: HashMap::new(),
        }
    }

    /// Marks everything in `listing` as already seen, so it is never reported unless it changes.
    pub fn mark_existing(&mut self, listing: &[Observation]) {
        for o in listing {
            self.reported.insert(o.path.clone(), o.stamp());
        }
    }

    /// Feeds one listing taken at `now`. Returns the files that have just become stable, each at
    /// most once per distinct size and modification time. Empty files are never reported.
    pub fn update(&mut self, now: Instant, listing: &[Observation]) -> Vec<Observation> {
        let present: HashSet<&Path> = listing.iter().map(|o| o.path.as_path()).collect();
        self.tracking.retain(|p, _| present.contains(p.as_path()));
        self.reported.retain(|p, _| present.contains(p.as_path()));

        let mut ready = Vec::new();
        for o in listing {
            if self.reported.get(&o.path) == Some(&o.stamp()) {
                continue;
            }
            let entry = self.tracking.get_mut(&o.path);
            match entry {
                Some(t) if t.stamp == o.stamp() => {
                    if o.size > 0 && now.saturating_duration_since(t.since) >= self.stable_for {
                        self.tracking.remove(&o.path);
                        self.reported.insert(o.path.clone(), o.stamp());
                        ready.push(o.clone());
                    }
                }
                _ => {
                    self.tracking.insert(
                        o.path.clone(),
                        Tracked {
                            stamp: o.stamp(),
                            since: now,
                        },
                    );
                }
            }
        }
        ready
    }
}

/// Lists the `.zip` files in `folder` whose names match, with their sizes. A folder that cannot be
/// read gives an error; an unreadable single entry is skipped.
///
/// # Errors
///
/// The error from reading the folder.
pub fn list_matching_zips(
    folder: &Path,
    matcher: &ZipNameMatcher,
) -> io::Result<Vec<(Observation, Vec<String>)>> {
    let mut out = Vec::new();
    for entry in fs::read_dir(folder)? {
        let Ok(entry) = entry else { continue };
        let name = entry.file_name().to_string_lossy().into_owned();
        let packages = matcher.packages_for(&name);
        if packages.is_empty() {
            continue;
        }
        let Ok(meta) = entry.metadata() else { continue };
        if !meta.is_file() {
            continue;
        }
        out.push((
            Observation {
                path: entry.path(),
                size: meta.len(),
                modified: meta.modified().ok(),
            },
            packages,
        ));
    }
    out.sort_by(|a, b| a.0.path.cmp(&b.0.path));
    Ok(out)
}

/// A running watch. Stops when dropped or when [`DownloadsWatcher::stop`] is called.
#[derive(Debug)]
pub struct DownloadsWatcher {
    stop: Arc<AtomicBool>,
    error: Arc<Mutex<Option<String>>>,
    handle: Option<JoinHandle<()>>,
}

impl DownloadsWatcher {
    /// Starts watching `folder` and calls `on_found` (from the watcher's thread) for each finished
    /// download that matches `matcher`.
    ///
    /// The folder is listed once before this returns, so a folder the app is not allowed to read
    /// (macOS asks for permission on first access) is reported here, not silently ignored.
    ///
    /// # Errors
    ///
    /// The error from the first listing of `folder`, or from starting the thread.
    pub fn start<F>(
        folder: impl Into<PathBuf>,
        matcher: ZipNameMatcher,
        config: &WatchConfig,
        mut on_found: F,
    ) -> io::Result<Self>
    where
        F: FnMut(FoundZip) + Send + 'static,
    {
        let folder = folder.into();
        let first = list_matching_zips(&folder, &matcher)?;
        let stop = Arc::new(AtomicBool::new(false));
        let error = Arc::new(Mutex::new(None));
        let config = config.clone();
        let (stop_flag, error_slot) = (Arc::clone(&stop), Arc::clone(&error));
        let handle = thread::Builder::new()
            .name("yellowmoon-watch-downloads".to_string())
            .spawn(move || {
                let mut tracker = StabilityTracker::new(config.stable_for);
                let mut packages: HashMap<PathBuf, Vec<String>> = HashMap::new();
                // `None` after a failed listing: the tracker then keeps what it knew, so zips that
                // were already seen or reported are not mistaken for new ones when the folder
                // comes back (access withdrawn and granted again, a volume remounted).
                let mut listing: Option<Vec<(Observation, Vec<String>)>> = Some(first);
                let mut first_round = true;
                while !stop_flag.load(Ordering::SeqCst) {
                    if let Some(current) = &listing {
                        let observed: Vec<Observation> =
                            current.iter().map(|(o, _)| o.clone()).collect();
                        for (o, p) in current {
                            packages.insert(o.path.clone(), p.clone());
                        }
                        if first_round && !config.report_existing {
                            tracker.mark_existing(&observed);
                        }
                        first_round = false;
                        for ready in tracker.update(Instant::now(), &observed) {
                            let ids = packages.get(&ready.path).cloned().unwrap_or_default();
                            on_found(FoundZip {
                                path: ready.path,
                                bytes: ready.size,
                                packages: ids,
                            });
                        }
                    }
                    // Sleep in short slices so `stop` is quick.
                    let wake = Instant::now() + config.poll_interval;
                    while Instant::now() < wake && !stop_flag.load(Ordering::SeqCst) {
                        thread::sleep(SLEEP_SLICE.min(config.poll_interval));
                    }
                    match list_matching_zips(&folder, &matcher) {
                        Ok(next) => {
                            listing = Some(next);
                            *error_slot.lock().unwrap_or_else(PoisonError::into_inner) = None;
                        }
                        Err(e) => {
                            listing = None;
                            *error_slot.lock().unwrap_or_else(PoisonError::into_inner) =
                                Some(e.to_string());
                        }
                    }
                }
            })?;
        Ok(Self {
            stop,
            error,
            handle: Some(handle),
        })
    }

    /// The error from the latest listing, if it failed (the folder was removed or access was
    /// withdrawn). Cleared by the next successful listing.
    #[must_use]
    pub fn last_error(&self) -> Option<String> {
        self.error
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// Stops the watcher and waits for its thread.
    pub fn stop(mut self) {
        self.shutdown();
    }

    fn shutdown(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for DownloadsWatcher {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obs(name: &str, size: u64, mtime_secs: u64) -> Observation {
        Observation {
            path: PathBuf::from(name),
            size,
            modified: Some(SystemTime::UNIX_EPOCH + Duration::from_secs(mtime_secs)),
        }
    }

    #[test]
    fn a_file_is_reported_once_after_two_quiet_seconds() {
        let t0 = Instant::now();
        let mut tracker = StabilityTracker::new(Duration::from_secs(2));
        let at = |s: f64| t0 + Duration::from_secs_f64(s);
        let file = obs("a.zip", 1000, 5);
        assert!(
            tracker
                .update(at(0.0), std::slice::from_ref(&file))
                .is_empty()
        );
        assert!(
            tracker
                .update(at(1.0), std::slice::from_ref(&file))
                .is_empty()
        );
        assert!(
            tracker
                .update(at(1.9), std::slice::from_ref(&file))
                .is_empty()
        );
        assert_eq!(
            tracker.update(at(2.0), std::slice::from_ref(&file)),
            vec![file.clone()]
        );
        // Never again while unchanged.
        assert!(
            tracker
                .update(at(2.5), std::slice::from_ref(&file))
                .is_empty()
        );
        assert!(tracker.update(at(60.0), &[file]).is_empty());
    }

    #[test]
    fn growing_files_restart_the_clock() {
        let t0 = Instant::now();
        let mut tracker = StabilityTracker::new(Duration::from_secs(2));
        let at = |s: f64| t0 + Duration::from_secs_f64(s);
        assert!(tracker.update(at(0.0), &[obs("a.zip", 10, 1)]).is_empty());
        assert!(tracker.update(at(1.5), &[obs("a.zip", 20, 2)]).is_empty());
        // 2.5 s after the first sight, but only 1 s since the last change.
        assert!(tracker.update(at(2.5), &[obs("a.zip", 20, 2)]).is_empty());
        assert!(tracker.update(at(3.4), &[obs("a.zip", 20, 2)]).is_empty());
        assert_eq!(
            tracker.update(at(3.5), &[obs("a.zip", 20, 2)]).len(),
            1,
            "two quiet seconds after the last change"
        );
    }

    #[test]
    fn a_redownload_under_the_same_name_is_reported_again() {
        let t0 = Instant::now();
        let mut tracker = StabilityTracker::new(Duration::from_secs(1));
        let at = |s: u64| t0 + Duration::from_secs(s);
        tracker.update(at(0), &[obs("a.zip", 10, 1)]);
        assert_eq!(tracker.update(at(1), &[obs("a.zip", 10, 1)]).len(), 1);
        // The file changes, settles, and is reported again.
        assert!(tracker.update(at(2), &[obs("a.zip", 30, 9)]).is_empty());
        assert_eq!(tracker.update(at(3), &[obs("a.zip", 30, 9)]).len(), 1);
    }

    #[test]
    fn existing_files_are_a_baseline_and_empty_files_wait() {
        let t0 = Instant::now();
        let mut tracker = StabilityTracker::new(Duration::from_secs(1));
        let old = obs("old.zip", 100, 1);
        tracker.mark_existing(std::slice::from_ref(&old));
        assert!(tracker.update(t0, std::slice::from_ref(&old)).is_empty());
        assert!(
            tracker
                .update(t0 + Duration::from_secs(5), std::slice::from_ref(&old))
                .is_empty()
        );
        let empty = obs("new.zip", 0, 2);
        tracker.update(t0, std::slice::from_ref(&empty));
        assert!(
            tracker
                .update(t0 + Duration::from_secs(5), std::slice::from_ref(&empty))
                .is_empty(),
            "an empty file is a download that has not started writing"
        );
    }

    #[test]
    fn deleted_files_are_forgotten() {
        let t0 = Instant::now();
        let mut tracker = StabilityTracker::new(Duration::from_secs(1));
        let f = obs("a.zip", 10, 1);
        tracker.update(t0, std::slice::from_ref(&f));
        tracker.update(t0 + Duration::from_secs(1), &[]);
        // Back with the same stamp after a gap: tracked afresh, so reported after a new quiet period.
        assert!(
            tracker
                .update(t0 + Duration::from_secs(2), std::slice::from_ref(&f))
                .is_empty()
        );
        assert_eq!(
            tracker
                .update(t0 + Duration::from_secs(3), std::slice::from_ref(&f))
                .len(),
            1
        );
    }

    #[test]
    fn matcher_needs_a_zip_and_a_package_pattern() {
        let db = PackageDb::from_json(
            r#"{"version":1,"packages":[
              {"id":"a","title":"A","role":"looks","input":"logc4",
               "zip_name":[{"all_of":["look","logc4"],"none_of":["logc3"]}]},
              {"id":"b","title":"B","role":"display_luts","input":"logc4",
               "zip_name":[{"all_of":["display","logc4"]}]}]}"#,
        )
        .unwrap();
        let all = ZipNameMatcher::new(&db, None);
        assert_eq!(all.packages_for("ARRI_Look_Library_LogC4.zip"), vec!["a"]);
        assert_eq!(
            all.packages_for("ARRI Display LUTs LogC4 (1).ZIP"),
            vec!["b"]
        );
        assert!(
            all.packages_for("ARRI_Look_Library_LogC4.zip.crdownload")
                .is_empty()
        );
        assert!(all.packages_for("ARRI_Look_Library_LogC4.pdf").is_empty());
        assert!(all.packages_for("holiday.zip").is_empty());
        assert!(all.packages_for("Look_Library_LogC3.zip").is_empty());
        let only_b = ZipNameMatcher::new(&db, Some(&["b".to_string()]));
        assert!(
            only_b
                .packages_for("ARRI_Look_Library_LogC4.zip")
                .is_empty()
        );
        assert_eq!(only_b.packages_for("display_logc4.zip"), vec!["b"]);
    }
}
