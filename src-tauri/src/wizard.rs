//! The ARRI import wizard's side of the shell (`docs/08` §Wizard, `docs/11` §Screen notes): what a
//! file the person chose is (`looks_recognise`), and the Downloads watcher (`looks_watch_downloads`).
//!
//! Recognition itself is `ym_looks::import::inspect`; this module turns its answer into the
//! `Recognition` of `src/api/types.ts`. The importer is the one `ym looks import` uses, so what
//! the wizard promises is what the import does.

use std::io;
use std::path::Path;

use ym_looks::import::inspect;
use ym_looks::library::LookFormat;
use ym_looks::recognize::{
    IssueKind, Outcome, PackageDb, PackageFinding, PackageStatus, Recognition as Found, Severity,
    Verdict,
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

/// "87 .cube · 65-point", or "87 .alf4 · 33-point" for the look-file version of a library.
fn contents_of(db: &PackageDb, finding: &PackageFinding) -> String {
    let noun = db
        .package(&finding.id)
        .map_or(LookFormat::Cube, |p| p.format)
        .noun();
    match finding.lut_sizes.as_slice() {
        [size] => format!("{} {noun} · {size}-point", finding.found),
        _ => format!("{} {noun}", finding.found),
    }
}

/// What the recogniser said about the files it refused while others in the same source are fine
/// ("Torn.alf4 cannot be used. Its checksum does not match, …", and how many more there are): the
/// row says them, so the person knows before the import which files will not arrive.
fn refused_notes(found: &Found) -> impl Iterator<Item = String> + '_ {
    found
        .issues
        .iter()
        .filter(|i| i.kind == IssueKind::Rejected)
        .map(|i| i.message.clone())
}

/// How many look files of the person's own the source holds.
fn own_look_files(found: &Found) -> u32 {
    u32::try_from(found.files.iter().filter(|f| f.is_own_look_file()).count()).unwrap_or(u32::MAX)
}

/// The row for look files of the person's own: "ARRI Look Files · LogC4", "3 .alf4 · 33-point".
/// `None` when the source holds no such file.
fn look_files_of(found: &Found) -> Option<Recognition> {
    let own: Vec<_> = found
        .files
        .iter()
        .filter(|f| matches!(f.outcome, Outcome::OwnLookFile { .. }))
        .collect();
    if own.is_empty() {
        return None;
    }
    let mut inputs: Vec<ym_color::look::LookInput> = Vec::new();
    for f in &own {
        if let Outcome::OwnLookFile { input, .. } = &f.outcome
            && !inputs.contains(input)
        {
            inputs.push(*input);
        }
    }
    let input = match inputs.as_slice() {
        [one] => Some(*one),
        _ => None,
    };
    let what = match input {
        Some(ym_color::look::LookInput::LogC4) => "ARRI Look Files · LogC4",
        Some(ym_color::look::LookInput::LogC3ToRec709) => "ARRI Look Files · LogC3 to Rec.709",
        _ => "ARRI Look Files",
    }
    .to_owned();
    let counts: Vec<String> = [LookFormat::Alf4, LookFormat::Alf4c, LookFormat::Aml]
        .into_iter()
        .filter_map(|format| {
            let n = own.iter().filter(|f| f.format == format).count();
            (n > 0).then(|| format!("{n} {}", format.noun()))
        })
        .collect();
    let mut contents = counts.join(" · ");
    let mut sizes: Vec<usize> = own.iter().filter_map(|f| f.lut_size).collect();
    sizes.sort_unstable();
    sizes.dedup();
    if let [size] = sizes.as_slice() {
        contents.push_str(&format!(" · {size}-point"));
    }
    let mut notes =
        vec!["Imported as your own looks, with the input read from each file.".to_owned()];
    notes.extend(refused_notes(found));
    notes.extend(
        found
            .issues
            .iter()
            .filter(|i| i.kind == IssueKind::DisplayTransformsIgnored)
            .map(|i| i.message.clone()),
    );
    Some(Recognition::LookFiles {
        what,
        contents,
        file_count: u32::try_from(own.len()).unwrap_or(u32::MAX),
        input: input.map(look_input),
        note: Some(notes.join(" ")),
    })
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
            let question = found
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
            let note = std::iter::once(question)
                .chain(refused_notes(found))
                .collect::<Vec<_>>()
                .join(" ");
            Recognition::NeedsAnswer {
                file_count: u32::try_from(asked.len()).unwrap_or(u32::MAX),
                lut_size,
                suggestion,
                look_file_count: own_look_files(found),
                note,
            }
        }
        Verdict::Ready => {
            let Some(finding) = found.packages.first() else {
                // No package: the person's own look files, if that is what the source holds.
                if let Some(files) = look_files_of(found) {
                    return files;
                }
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
            notes.extend(refused_notes(found));
            let own = found.files.iter().filter(|f| f.is_own_look_file()).count();
            if own > 0 {
                notes.push(format!(
                    "It also holds {} of your own, imported as your own looks.",
                    if own == 1 {
                        "1 look file".to_owned()
                    } else {
                        format!("{own} look files")
                    }
                ));
            }
            Recognition::Package {
                package_id,
                title: finding.title.clone(),
                what: what_of(db, finding),
                contents: contents_of(db, finding),
                file_count: u32::try_from(finding.found).unwrap_or(u32::MAX),
                look_file_count: own_look_files(found),
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

#[cfg(test)]
mod tests {
    use super::*;
    use ym_color::alf::AlfFile;
    use ym_color::lut::Lut3d;

    /// One small package, with the id and the names of ARRI's LogC4 look library.
    fn db() -> PackageDb {
        PackageDb::from_json(
            r#"{
            "version": 1,
            "packages": [
              { "id": "look-library-logc4", "title": "Look Library · LogC4", "role": "looks",
                "input": "logc4",
                "zip_name": [ { "all_of": ["look", "logc4"], "none_of": ["logc3"] } ],
                "expected": { "cube_files": 3, "lut_sizes": [5] },
                "markers": { "encoding": { "any_of": ["logc4"], "none_of": ["logc3"] },
                             "display_rendering": "absent", "group_folders": "required" } }
            ] }"#,
        )
        .expect("package list")
    }

    fn write_cube(dir: &Path, name: &str, title: &str) {
        std::fs::create_dir_all(dir).expect("folder");
        let text = Lut3d::identity(5).expect("identity").to_cube_string(title);
        std::fs::write(dir.join(name), text).expect("cube");
    }

    /// An `.alf4` from our own writer; with `damaged`, one byte of it flipped.
    fn write_alf4(dir: &Path, name: &str, damaged: bool) {
        std::fs::create_dir_all(dir).expect("folder");
        let lut = Lut3d::identity(5).expect("identity");
        let mut bytes = AlfFile::alf4(&lut, None, "", "x.cube")
            .to_bytes()
            .expect("write");
        if damaged {
            let at = bytes.len() - 7;
            bytes[at] ^= 0x08;
        }
        std::fs::write(dir.join(name), bytes).expect("look file");
    }

    const TORN: &str = "Torn.alf4 cannot be used. Its checksum does not match, so the file is damaged. Download it again.";

    #[test]
    fn a_folder_of_look_files_names_the_one_that_cannot_be_used() {
        let dir = tempfile::tempdir().expect("dir");
        let folder = dir.path().join("Mine");
        write_alf4(&folder, "Warm.alf4", false);
        write_alf4(&folder, "Cool.alf4", false);
        write_alf4(&folder, "Torn.alf4", true);
        let seen = recognise(&db(), &folder);
        match seen.recognition {
            Recognition::LookFiles {
                file_count: 2,
                note: Some(note),
                ..
            } => {
                assert!(note.contains(TORN), "{note}");
                assert!(note.contains("Imported as your own looks"), "{note}");
            }
            other => panic!("expected two look files and a note, got {other:?}"),
        }
        // Without a damaged file the note says nothing about one.
        std::fs::remove_file(folder.join("Torn.alf4")).expect("remove");
        match recognise(&db(), &folder).recognition {
            Recognition::LookFiles {
                note: Some(note), ..
            } => assert!(!note.contains("cannot be used"), "{note}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_package_folder_names_a_refused_file_and_counts_the_look_files_it_also_holds() {
        let dir = tempfile::tempdir().expect("dir");
        let folder = dir.path().join("Look_Library_LogC4");
        for (group, name) in [
            ("3000 Film", "Film_Warm_LogC4.cube"),
            ("3000 Film", "Film_Cool_LogC4.cube"),
            ("4000 Mono", "Mono_LogC4.cube"),
        ] {
            write_cube(&folder.join(group), name, name);
        }
        write_alf4(&folder, "Mine.alf4", false);
        write_alf4(&folder, "Torn.alf4", true);
        let seen = recognise(&db(), &folder);
        match seen.recognition {
            Recognition::Package {
                file_count: 3,
                look_file_count: 1,
                note: Some(note),
                ..
            } => {
                assert!(note.contains(TORN), "{note}");
                assert!(
                    note.contains("It also holds 1 look file of your own"),
                    "{note}"
                );
            }
            other => panic!("expected the package with one look file, got {other:?}"),
        }
        // A package alone has no look files and no refusals.
        std::fs::remove_file(folder.join("Mine.alf4")).expect("remove");
        std::fs::remove_file(folder.join("Torn.alf4")).expect("remove");
        match recognise(&db(), &folder).recognition {
            Recognition::Package {
                look_file_count: 0,
                note,
                ..
            } => assert!(note.is_none(), "{note:?}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_question_names_a_refused_file_and_counts_the_look_files_beside_it() {
        let dir = tempfile::tempdir().expect("dir");
        let folder = dir.path().join("Generic");
        write_cube(&folder, "Good.cube", "Good");
        std::fs::write(folder.join("Torn.cube"), "LUT_3D_SIZE 5\n0 0 0\n").expect("torn");
        write_alf4(&folder, "Own.alf4", false);
        match recognise(&db(), &folder).recognition {
            Recognition::NeedsAnswer {
                file_count: 1,
                look_file_count: 1,
                note,
                ..
            } => {
                assert!(note.contains("Torn.cube cannot be used."), "{note}");
                assert!(note.contains("Say what"), "the question stays: {note}");
            }
            other => panic!("expected a question, got {other:?}"),
        }
    }
}
