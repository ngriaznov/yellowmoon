//! Reading the thing the user chose: one `.cube` file, a folder, or a zip.
//!
//! Listing never reads LUT data. A zip is opened through its central directory and entries are
//! inflated one at a time, so a 180 MB look package never sits in memory whole. Only `.cube` files
//! are candidates; `__MACOSX` folders, `._` files, `.DS_Store`, `.alf4`, `.alf2`, PDFs and
//! everything else are counted in [`Ignored`] and left alone. Nothing here writes.

use std::fs::{self, File};
use std::io::{self, BufReader, Read};
use std::path::{Component, Path, PathBuf};

use ym_color::cube::MAX_FILE_BYTES;
use zip::ZipArchive;
use zip::result::ZipError;

use super::ImportError;
use crate::recognize::names::file_name;
use crate::recognize::{Ignored, SourceInfo, SourceKind};

/// Most files or zip entries a source may hold before it is refused as not being a look package.
pub const MAX_ENTRIES: usize = 100_000;
/// Deepest folder nesting that is walked.
const MAX_DEPTH: usize = 16;
/// Most `.cube` data one source may hold or be read for, in bytes. The biggest ARRI package is
/// under a gigabyte; this stops a hostile zip (small on disk, enormous when inflated) early.
pub const MAX_TOTAL_BYTES: u64 = 4 << 30;

/// A `.cube` file inside a source.
#[derive(Debug, Clone)]
pub(crate) struct Candidate {
    /// `/`-separated path inside the source.
    pub(crate) rel_path: String,
    /// Uncompressed size in bytes.
    pub(crate) size: u64,
    locator: Locator,
}

#[derive(Debug, Clone)]
enum Locator {
    Path(PathBuf),
    ZipIndex(usize),
}

/// Why a candidate could not be read.
#[derive(Debug)]
pub(crate) enum ReadFailure {
    /// Over the `.cube` size limit.
    TooLarge,
    /// An I/O or decompression error, as a sentence.
    Failed(String),
}

impl ReadFailure {
    /// A sentence for the person importing.
    pub(crate) fn sentence(&self) -> String {
        match self {
            Self::TooLarge => format!(
                "It is larger than the {} MB limit for a .cube file.",
                MAX_FILE_BYTES / (1024 * 1024)
            ),
            Self::Failed(reason) => format!("It could not be read: {reason}."),
        }
    }
}

enum Opened {
    File,
    Folder,
    Zip(Box<ZipArchive<BufReader<File>>>),
}

/// An opened source with its `.cube` candidates listed.
pub(crate) struct Source {
    pub(crate) info: SourceInfo,
    pub(crate) candidates: Vec<Candidate>,
    pub(crate) ignored: Ignored,
    opened: Opened,
    /// Bytes this source may still be read for; see [`MAX_TOTAL_BYTES`].
    budget: u64,
}

/// What a name inside a source is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EntryKind {
    Cube,
    Alf4,
    Alf2,
    Pdf,
    MacMetadata,
    Other,
}

fn classify(rel_path: &str) -> EntryKind {
    let name = file_name(rel_path);
    if rel_path.split('/').any(|c| c == "__MACOSX") || name.starts_with("._") || name == ".DS_Store"
    {
        return EntryKind::MacMetadata;
    }
    match name
        .rsplit_once('.')
        .map(|(_, ext)| ext.to_ascii_lowercase())
    {
        Some(ext) if ext == "cube" => EntryKind::Cube,
        Some(ext) if ext == "alf4" => EntryKind::Alf4,
        Some(ext) if ext == "alf2" => EntryKind::Alf2,
        Some(ext) if ext == "pdf" => EntryKind::Pdf,
        _ => EntryKind::Other,
    }
}

fn tally(ignored: &mut Ignored, kind: EntryKind) {
    match kind {
        EntryKind::Alf4 => ignored.alf4 += 1,
        EntryKind::Alf2 => ignored.alf2 += 1,
        EntryKind::Pdf => ignored.pdf += 1,
        EntryKind::MacMetadata => ignored.macos_metadata += 1,
        EntryKind::Other => ignored.other += 1,
        EntryKind::Cube => {}
    }
}

fn is_zip(path: &Path) -> io::Result<bool> {
    let by_name = path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("zip"));
    if by_name {
        return Ok(true);
    }
    let mut magic = [0_u8; 4];
    let mut file = File::open(path)?;
    let n = file.read(&mut magic)?;
    Ok(n == 4 && magic == *b"PK\x03\x04")
}

fn zip_error(error: &ZipError) -> ImportError {
    ImportError::BadZip(match error {
        ZipError::InvalidArchive(_) | ZipError::Io(_) => {
            "It is damaged or was not downloaded completely".to_string()
        }
        ZipError::UnsupportedArchive(reason) => {
            format!("It uses a feature we cannot read ({reason})")
        }
        other => other.to_string(),
    })
}

fn display_name(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .or_else(|| {
            fs::canonicalize(path)
                .ok()
                .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
        })
        .unwrap_or_else(|| "folder".to_string())
}

impl Source {
    /// Opens `path` and lists its `.cube` candidates.
    pub(crate) fn open(path: &Path) -> Result<Self, ImportError> {
        let source = Self::open_unchecked(path)?;
        if declared_total(&source.candidates) > MAX_TOTAL_BYTES {
            return Err(ImportError::TooLarge);
        }
        Ok(source)
    }

    fn open_unchecked(path: &Path) -> Result<Self, ImportError> {
        let meta = match fs::metadata(path) {
            Ok(m) => m,
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                return Err(ImportError::SourceMissing(path.to_path_buf()));
            }
            Err(source) => {
                return Err(ImportError::Io {
                    path: path.to_path_buf(),
                    source,
                });
            }
        };
        let name = display_name(path);
        let io_error = |source| ImportError::Io {
            path: path.to_path_buf(),
            source,
        };
        if meta.is_dir() {
            let mut candidates = Vec::new();
            let mut ignored = Ignored::default();
            let mut count = 0;
            // Paths are relative to the chosen folder, as they are inside a zip: the folder's own
            // name is the user's, not part of the package, and is evidence only through
            // `SourceInfo::name`. (A folder called "2026-10-12 downloads" is not a look group.)
            walk(path, "", 0, &mut candidates, &mut ignored, &mut count)?;
            candidates.sort_by(|a, b| a.rel_path.cmp(&b.rel_path));
            return Ok(Self {
                info: SourceInfo {
                    kind: SourceKind::Folder,
                    name,
                },
                candidates,
                ignored,
                opened: Opened::Folder,
                budget: MAX_TOTAL_BYTES,
            });
        }
        if is_zip(path).map_err(io_error)? {
            let file = File::open(path).map_err(io_error)?;
            let mut archive = ZipArchive::new(BufReader::new(file)).map_err(|e| zip_error(&e))?;
            let (candidates, ignored) = list_zip(&mut archive)?;
            return Ok(Self {
                info: SourceInfo {
                    kind: SourceKind::Zip,
                    name,
                },
                candidates,
                ignored,
                opened: Opened::Zip(Box::new(archive)),
                budget: MAX_TOTAL_BYTES,
            });
        }
        if classify(&name) == EntryKind::Cube {
            let candidate = Candidate {
                rel_path: name.clone(),
                size: meta.len(),
                locator: Locator::Path(path.to_path_buf()),
            };
            return Ok(Self {
                info: SourceInfo {
                    kind: SourceKind::File,
                    name,
                },
                candidates: vec![candidate],
                ignored: Ignored::default(),
                opened: Opened::File,
                budget: MAX_TOTAL_BYTES,
            });
        }
        Err(ImportError::UnsupportedSource(name))
    }

    /// Reads one candidate completely (at most [`MAX_FILE_BYTES`]).
    pub(crate) fn read(&mut self, index: usize) -> Result<Vec<u8>, ReadFailure> {
        let candidate = self
            .candidates
            .get(index)
            .cloned()
            .ok_or_else(|| ReadFailure::Failed("it is not in the source".to_string()))?;
        if candidate.size > MAX_FILE_BYTES as u64 {
            return Err(ReadFailure::TooLarge);
        }
        if self.budget == 0 {
            return Err(ReadFailure::Failed(
                "this source holds far more data than a set of looks".to_string(),
            ));
        }
        let limit = MAX_FILE_BYTES as u64 + 1;
        let capacity = usize::try_from(candidate.size).unwrap_or(0).min(64 << 20);
        let mut bytes = Vec::with_capacity(capacity);
        let failed = |e: io::Error| ReadFailure::Failed(e.to_string());
        match (&candidate.locator, &mut self.opened) {
            (Locator::Path(path), _) => {
                File::open(path)
                    .and_then(|f| f.take(limit).read_to_end(&mut bytes))
                    .map_err(failed)?;
            }
            (Locator::ZipIndex(i), Opened::Zip(archive)) => {
                let entry = archive.by_index(*i).map_err(|e| match e {
                    ZipError::UnsupportedArchive(reason) => {
                        ReadFailure::Failed(format!("the zip entry cannot be opened ({reason})"))
                    }
                    other => ReadFailure::Failed(other.to_string()),
                })?;
                entry.take(limit).read_to_end(&mut bytes).map_err(failed)?;
            }
            (Locator::ZipIndex(_), _) => {
                return Err(ReadFailure::Failed("the source is not a zip".to_string()));
            }
        }
        // Count what was inflated even when it is refused: an entry that claims to be small and
        // inflates past the limit must use up the budget, or a zip of them would be read forever.
        self.budget = self.budget.saturating_sub(bytes.len() as u64);
        if bytes.len() as u64 > MAX_FILE_BYTES as u64 {
            return Err(ReadFailure::TooLarge);
        }
        Ok(bytes)
    }
}

/// The sizes the candidates declare, added up without overflow: a hostile zip can declare sizes
/// near `u64::MAX` for several entries, and the sum must come out huge, not wrap or panic.
fn declared_total(candidates: &[Candidate]) -> u64 {
    candidates
        .iter()
        .fold(0_u64, |total, c| total.saturating_add(c.size))
}

fn list_zip(
    archive: &mut ZipArchive<BufReader<File>>,
) -> Result<(Vec<Candidate>, Ignored), ImportError> {
    if archive.len() > MAX_ENTRIES {
        return Err(ImportError::BadZip(format!(
            "It has {} entries, far more than a look package",
            archive.len()
        )));
    }
    let mut candidates = Vec::new();
    let mut ignored = Ignored::default();
    for index in 0..archive.len() {
        let entry = archive.by_index_raw(index).map_err(|e| zip_error(&e))?;
        if entry.is_dir() {
            continue;
        }
        let safe = if entry.is_symlink() {
            None
        } else {
            entry.enclosed_name().and_then(|p| {
                let parts: Option<Vec<String>> = p
                    .components()
                    .map(|c| match c {
                        Component::Normal(s) => Some(s.to_string_lossy().into_owned()),
                        _ => None,
                    })
                    .collect();
                parts.map(|p| p.join("/"))
            })
        };
        let Some(rel_path) = safe else {
            ignored.unsafe_entries += 1;
            continue;
        };
        match classify(&rel_path) {
            EntryKind::Cube => candidates.push(Candidate {
                rel_path,
                size: entry.size(),
                locator: Locator::ZipIndex(index),
            }),
            other => tally(&mut ignored, other),
        }
    }
    candidates.sort_by(|a, b| a.rel_path.cmp(&b.rel_path));
    Ok((candidates, ignored))
}

fn walk(
    dir: &Path,
    prefix: &str,
    depth: usize,
    candidates: &mut Vec<Candidate>,
    ignored: &mut Ignored,
    count: &mut usize,
) -> Result<(), ImportError> {
    let io_error = |source| ImportError::Io {
        path: dir.to_path_buf(),
        source,
    };
    let mut entries: Vec<_> = fs::read_dir(dir)
        .map_err(io_error)?
        .collect::<Result<_, _>>()
        .map_err(io_error)?;
    entries.sort_by_key(fs::DirEntry::file_name);
    for entry in entries {
        *count += 1;
        if *count > MAX_ENTRIES {
            return Err(ImportError::TooManyFiles);
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        let rel_path = if prefix.is_empty() {
            name.clone()
        } else {
            format!("{prefix}/{name}")
        };
        let kind = entry.file_type().map_err(io_error)?;
        if kind.is_symlink() {
            ignored.unsafe_entries += 1;
        } else if kind.is_dir() {
            if name == "__MACOSX" {
                ignored.macos_metadata += 1;
            } else if depth < MAX_DEPTH {
                walk(
                    &entry.path(),
                    &rel_path,
                    depth + 1,
                    candidates,
                    ignored,
                    count,
                )?;
            } else {
                ignored.other += 1;
            }
        } else if kind.is_file() {
            match classify(&rel_path) {
                EntryKind::Cube => candidates.push(Candidate {
                    size: entry.metadata().map_err(io_error)?.len(),
                    rel_path,
                    locator: Locator::Path(entry.path()),
                }),
                other => tally(ignored, other),
            }
        } else {
            ignored.other += 1;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reading_stops_when_the_budget_is_spent() {
        let dir = tempfile::tempdir().unwrap();
        for name in ["a.cube", "b.cube", "c.cube"] {
            fs::write(dir.path().join(name), "x".repeat(100)).unwrap();
        }
        let mut source = Source::open(dir.path()).unwrap();
        assert_eq!(source.candidates.len(), 3);
        source.budget = 150;
        assert!(source.read(0).is_ok());
        assert!(
            source.read(1).is_ok(),
            "the read that crosses the budget still completes"
        );
        match source.read(2) {
            Err(ReadFailure::Failed(reason)) => {
                assert!(reason.contains("far more data"), "{reason}")
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_file_that_inflates_past_the_limit_still_uses_up_the_budget() {
        // A sparse file reads as zeros but is bigger than a .cube file may be, while the listing
        // says it is small: what a zip entry with a lying size looks like to `read`.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("liar.cube");
        let file = File::create(&path).unwrap();
        file.set_len(MAX_FILE_BYTES as u64 + 4096).unwrap();
        drop(file);
        let mut source = Source {
            info: SourceInfo {
                kind: SourceKind::File,
                name: "liar.cube".to_string(),
            },
            candidates: vec![Candidate {
                rel_path: "liar.cube".to_string(),
                size: 10,
                locator: Locator::Path(path),
            }],
            ignored: Ignored::default(),
            opened: Opened::File,
            budget: 1000,
        };
        assert!(matches!(source.read(0), Err(ReadFailure::TooLarge)));
        assert_eq!(source.budget, 0, "the refused read must be counted");
        match source.read(0) {
            Err(ReadFailure::Failed(reason)) => assert!(reason.contains("far more data")),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn declared_sizes_add_up_without_overflow() {
        let candidate = |size| Candidate {
            rel_path: "a.cube".to_string(),
            size,
            locator: Locator::ZipIndex(0),
        };
        // Two entries that each claim to inflate to 16 exabytes.
        let total = declared_total(&[candidate(u64::MAX), candidate(u64::MAX), candidate(7)]);
        assert_eq!(total, u64::MAX);
        assert!(total > MAX_TOTAL_BYTES);
        assert_eq!(declared_total(&[]), 0);
        assert_eq!(declared_total(&[candidate(5), candidate(6)]), 11);
    }

    #[test]
    fn names_are_classified() {
        assert_eq!(classify("a/b.cube"), EntryKind::Cube);
        assert_eq!(classify("a/B.CUBE"), EntryKind::Cube);
        assert_eq!(classify("x.alf4"), EntryKind::Alf4);
        assert_eq!(classify("x.ALF2"), EntryKind::Alf2);
        assert_eq!(classify("doc/readme.pdf"), EntryKind::Pdf);
        assert_eq!(classify("__MACOSX/a/b.cube"), EntryKind::MacMetadata);
        assert_eq!(classify("a/._b.cube"), EntryKind::MacMetadata);
        assert_eq!(classify("a/.DS_Store"), EntryKind::MacMetadata);
        assert_eq!(classify("a/notes.txt"), EntryKind::Other);
        assert_eq!(classify("a/noextension"), EntryKind::Other);
    }
}
