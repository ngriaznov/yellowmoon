//! The ARRI import wizard's side of the shell (`docs/08` §Wizard, `docs/11` §Screen notes): what a
//! file the person chose is (`looks_recognise`), and the Downloads watcher (`looks_watch_downloads`).
//!
//! Recognition itself is `ym_looks::import::inspect`; this module turns its answer into the
//! `Recognition` of `src/api/types.ts`. The importer is the one `ym looks import` uses, so what
//! the wizard promises is what the import does.

use std::io;
use std::path::Path;

use ym_looks::import::inspect;
use ym_looks::recognize::{
    Outcome, PackageDb, PackageFinding, PackageStatus, Recognition as Found, Severity, Verdict,
};

use crate::events::{AppEvent, Emitter};
use crate::wire::{
    DownloadFound, LookInput, Num, PackageId, Recognition, SourceKind, SourceRecognition,
    WatchStatus,
};

use std::sync::Arc;
use ym_looks::import::watch::{DownloadsWatcher, FoundZip, WatchConfig, ZipNameMatcher};

/// The kind of source a path is, from the disk or, for a path that is not there, its name.
fn source_kind(path: &Path) -> SourceKind {
    match std::fs::metadata(path) {
        Ok(m) if m.is_dir() => SourceKind::Folder,
        _ => {
            let ext = path
                .extension()
                .map(|e| e.to_string_lossy().to_ascii_lowercase());
            match ext.as_deref() {
                Some("zip") => SourceKind::Zip,
                Some("cube") => SourceKind::File,
                _ if path.is_dir() => SourceKind::Folder,
                _ => SourceKind::File,
            }
        }
    }
}

fn look_input(input: ym_color::look::LookInput) -> LookInput {
    match input {
        ym_color::look::LookInput::LogC4 => LookInput::Logc4,
        ym_color::look::LookInput::LogC3Ei800 => LookInput::Logc3Ei800,
        ym_color::look::LookInput::LogC3ToRec709 => LookInput::Logc3ToRec709,
    }
}

/// What a source is. Never fails: a source that cannot be read is `unusable` or `missing`.
#[must_use]
pub fn recognise(db: &PackageDb, path: &Path) -> SourceRecognition {
    let name = path.file_name().map_or_else(
        || path.to_string_lossy().into_owned(),
        |n| n.to_string_lossy().into_owned(),
    );
    let source = source_kind(path);
    let reply = |size_bytes: Option<Num>, recognition: Recognition| SourceRecognition {
        path: path.to_string_lossy().into_owned(),
        name: name.clone(),
        source,
        size_bytes,
        recognition,
    };
    let meta = match std::fs::metadata(path) {
        Ok(meta) => meta,
        Err(_) => {
            return reply(
                None,
                Recognition::Missing {
                    reason: format!(
                        "{name} isn't there any more. Download it again, or choose it again."
                    ),
                },
            );
        }
    };
    let size_bytes = (!meta.is_dir()).then_some(Num(meta.len() as f64));
    match inspect(path, db, None, &mut |_| {}) {
        Ok(found) => reply(size_bytes, map(db, &found)),
        Err(e) => reply(
            size_bytes,
            Recognition::Unusable {
                reason: e.to_string(),
                package_id: None,
            },
        ),
    }
}

/// The wording of a package for the wizard's row: "Look Library · LogC4 · log to log".
fn what_of(db: &PackageDb, finding: &PackageFinding) -> String {
    let log_to_log = db
        .packages
        .iter()
        .find(|p| p.id == finding.id)
        .is_some_and(|p| {
            p.look_input().is_some_and(|i| {
                matches!(
                    i,
                    ym_color::look::LookInput::LogC4 | ym_color::look::LookInput::LogC3Ei800
                )
            })
        });
    if log_to_log {
        format!("{} · log to log", finding.title)
    } else {
        finding.title.clone()
    }
}

/// "87 .cube · 65-point".
fn contents_of(finding: &PackageFinding) -> String {
    match finding.lut_sizes.as_slice() {
        [size] => format!("{} .cube · {size}-point", finding.found),
        _ => format!("{} .cube", finding.found),
    }
}

fn map(db: &PackageDb, found: &Found) -> Recognition {
    let first_problem = || {
        found
            .problems()
            .next()
            .or_else(|| found.issues.first())
            .map(|i| i.message.clone())
    };
    match found.verdict() {
        Verdict::Blocked => Recognition::Unusable {
            reason: first_problem().unwrap_or_else(|| {
                "There is nothing in this that Yellowmoon can import.".to_owned()
            }),
            package_id: found
                .packages
                .first()
                .and_then(|p| PackageId::from_data_id(&p.id)),
        },
        Verdict::Incomplete => {
            let finding = found
                .packages
                .iter()
                .find(|p| p.status == PackageStatus::Incomplete);
            Recognition::Unusable {
                reason: found
                    .issues
                    .iter()
                    .find(|i| i.severity == Severity::Problem)
                    .map(|i| i.message.clone())
                    .or_else(first_problem)
                    .unwrap_or_else(|| {
                        "Some files of this package are missing. Download it again.".to_owned()
                    }),
                package_id: finding.and_then(|p| PackageId::from_data_id(&p.id)),
            }
        }
        Verdict::NeedsAnswer => {
            let asked: Vec<_> = found.files.iter().filter(|f| f.needs_answer()).collect();
            let sizes: Vec<usize> = asked.iter().filter_map(|f| f.lut_size).collect();
            let lut_size = match sizes.as_slice() {
                [first, rest @ ..] if rest.iter().all(|s| s == first) => u32::try_from(*first).ok(),
                _ => None,
            };
            let suggestion = asked.iter().find_map(|f| match &f.outcome {
                Outcome::NeedsAnswer {
                    suggested_input, ..
                } => suggested_input.map(look_input),
                _ => None,
            });
            let note = found
                .issues
                .iter()
                .find(|i| i.severity == Severity::Question)
                .map(|i| i.message.clone())
                .or_else(|| {
                    asked.iter().find_map(|f| match &f.outcome {
                        Outcome::NeedsAnswer { note, .. } => note.clone(),
                        _ => None,
                    })
                })
                .unwrap_or_else(|| {
                    "This isn't one of ARRI's packages. Say what the LUT expects as input."
                        .to_owned()
                });
            Recognition::NeedsAnswer {
                file_count: u32::try_from(asked.len()).unwrap_or(u32::MAX),
                lut_size,
                suggestion,
                note,
            }
        }
        Verdict::Ready => {
            let Some(finding) = found.packages.first() else {
                return Recognition::Unusable {
                    reason: first_problem().unwrap_or_else(|| {
                        "There is nothing in this that Yellowmoon can import.".to_owned()
                    }),
                    package_id: None,
                };
            };
            let Some(package_id) = PackageId::from_data_id(&finding.id) else {
                return Recognition::Unusable {
                    reason: format!(
                        "{} isn't a package this version of Yellowmoon knows. Update the app, then try again.",
                        finding.title
                    ),
                    package_id: None,
                };
            };
            // More than one package in one source, or more files than expected, is a remark and not
            // a reason to stop.
            let mut notes: Vec<String> = found
                .issues
                .iter()
                .filter(|i| i.severity == Severity::Info)
                .map(|i| i.message.clone())
                .collect();
            if found.packages.len() > 1 {
                notes.push(format!(
                    "This also holds {} other package{}.",
                    found.packages.len() - 1,
                    if found.packages.len() == 2 { "" } else { "s" }
                ));
            }
            Recognition::Package {
                package_id,
                title: finding.title.clone(),
                what: what_of(db, finding),
                contents: contents_of(finding),
                file_count: u32::try_from(finding.found).unwrap_or(u32::MAX),
                note: (!notes.is_empty()).then(|| notes.join(" ")),
            }
        }
    }
}

/// The plain words for a Downloads folder that cannot be read.
fn problem_of(error: &io::Error) -> String {
    if error.kind() == io::ErrorKind::PermissionDenied {
        "macOS didn't let Yellowmoon look in your Downloads folder. Allow it in System Settings, under Privacy & Security, Files and Folders, or choose the file yourself.".to_owned()
    } else {
        "Yellowmoon couldn't look in your Downloads folder. Choose the file yourself.".to_owned()
    }
}

/// The running Downloads watcher, if any (`looks_watch_downloads`).
#[derive(Default)]
pub struct Downloads {
    watcher: std::sync::Mutex<Option<DownloadsWatcher>>,
}

impl Downloads {
    /// Starts or stops watching `folder` for the zips of the packages in `db`. A zip that has
    /// stopped growing arrives as a `download_found` event.
    pub fn set(
        &self,
        enabled: bool,
        folder: Option<&Path>,
        db: &PackageDb,
        config: &WatchConfig,
        emitter: Arc<dyn Emitter>,
    ) -> WatchStatus {
        let mut slot = self
            .watcher
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !enabled {
            if let Some(w) = slot.take() {
                w.stop();
            }
            return WatchStatus {
                watching: false,
                problem: None,
            };
        }
        if slot.is_some() {
            return WatchStatus {
                watching: true,
                problem: None,
            };
        }
        let Some(folder) = folder else {
            return WatchStatus {
                watching: false,
                problem: Some(
                    "Yellowmoon couldn't find your Downloads folder. Choose the file yourself."
                        .to_owned(),
                ),
            };
        };
        let started = DownloadsWatcher::start(
            folder.to_path_buf(),
            ZipNameMatcher::new(db, None),
            config,
            move |zip: FoundZip| {
                let file_name = zip
                    .path
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default();
                let package_hint = match zip.packages.as_slice() {
                    [one] => PackageId::from_data_id(one),
                    _ => None,
                };
                emitter.emit(AppEvent::DownloadFound(DownloadFound {
                    path: zip.path.to_string_lossy().into_owned(),
                    file_name,
                    size_bytes: Num(zip.bytes as f64),
                    package_hint,
                }));
            },
        );
        match started {
            Ok(w) => {
                *slot = Some(w);
                WatchStatus {
                    watching: true,
                    problem: None,
                }
            }
            Err(e) => WatchStatus {
                watching: false,
                problem: Some(problem_of(&e)),
            },
        }
    }
}
