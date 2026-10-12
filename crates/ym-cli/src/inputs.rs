//! The photos a command was asked to work on: files, folders (walked for `*.ARW` in any case, one
//! level unless `--recursive`) and `-` for paths on stdin.
//!
//! The rules are those of `ym_core::session::Session::open`: names that start with a dot are
//! ignored (macOS leaves `._DSC0001.ARW` resource forks on exFAT cards), other file types inside a
//! folder are ignored, and the same file reached twice is one photo. A path the user named
//! explicitly that cannot be a photo (it does not exist, it is not an ARW file) is not ignored: it
//! comes back as a [`Problem`], so the run can say so and count it.

use std::collections::HashSet;
use std::io::{self, BufRead};
use std::path::{Path, PathBuf};

use ym_raw::error::RawError;

/// An input the user named that cannot be developed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Problem {
    /// The path as it was given.
    pub path: PathBuf,
    /// A short machine-readable kind (`not_found`, `not_arw`).
    pub kind: &'static str,
    /// What is wrong and what to do next.
    pub message: String,
}

/// The inputs after folders have been walked.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Expanded {
    /// The ARW files, as absolute paths in the order a session lists them (by path, ignoring case),
    /// each once.
    pub files: Vec<PathBuf>,
    /// Named paths that cannot be photos, in the order they were given.
    pub problems: Vec<Problem>,
    /// Entries inside folders that were passed over (not ARW, or hidden).
    pub ignored: usize,
}

/// Reads file paths from `reader`, one per line. Blank lines are skipped and a trailing carriage
/// return (a list made on Windows) is dropped.
pub fn read_paths(reader: impl BufRead) -> io::Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let line = line.trim_end_matches('\r');
        if !line.trim().is_empty() {
            out.push(PathBuf::from(line));
        }
    }
    Ok(out)
}

/// Replaces a `-` among the inputs by the paths `stdin` holds.
///
/// # Errors
///
/// The read error of `stdin`.
pub fn with_stdin(inputs: &[PathBuf], stdin: impl BufRead) -> io::Result<Vec<PathBuf>> {
    let mut stdin = Some(stdin);
    let mut out = Vec::new();
    for input in inputs {
        if input.as_os_str() == "-" {
            // A second `-` finds the list already read.
            if let Some(reader) = stdin.take() {
                out.extend(read_paths(reader)?);
            }
        } else {
            out.push(input.clone());
        }
    }
    Ok(out)
}

fn is_arw(path: &Path) -> bool {
    path.extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("arw"))
}

fn is_hidden(path: &Path) -> bool {
    path.file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|n| n.starts_with('.'))
}

fn walk(dir: &Path, recursive: bool, files: &mut Vec<PathBuf>, ignored: &mut usize) {
    let Ok(read) = std::fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<PathBuf> = read.filter_map(|e| e.ok().map(|e| e.path())).collect();
    entries.sort();
    for path in entries {
        let Ok(meta) = std::fs::symlink_metadata(&path) else {
            continue;
        };
        let is_file = meta.is_file() || (meta.file_type().is_symlink() && path.is_file());
        if is_hidden(&path) {
            *ignored += 1;
        } else if meta.is_dir() {
            if recursive {
                walk(&path, recursive, files, ignored);
            }
        } else if is_file && is_arw(&path) {
            files.push(path);
        } else {
            *ignored += 1;
        }
    }
}

fn canonical(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

/// The message for a file that is not an ARW file: what `probe` says about it, else the general
/// sentence.
fn not_arw_message(path: &Path) -> String {
    match ym_raw::metadata::probe(path) {
        Err(error) => error.to_string(),
        Ok(_) => RawError::NotArw.to_string(),
    }
}

/// Walks the inputs. Relative paths are taken against the working directory and returned absolute
/// (symbolic links are not resolved: a photo keeps the name it was given).
pub fn expand(inputs: &[PathBuf], recursive: bool) -> Expanded {
    let mut files: Vec<PathBuf> = Vec::new();
    let mut out = Expanded::default();
    for input in inputs {
        if input.is_dir() {
            walk(input, recursive, &mut files, &mut out.ignored);
        } else if input.is_file() {
            if is_arw(input) {
                files.push(input.clone());
            } else {
                out.problems.push(Problem {
                    path: input.clone(),
                    kind: "not_arw",
                    message: not_arw_message(input),
                });
            }
        } else {
            out.problems.push(Problem {
                path: input.clone(),
                kind: "not_found",
                message: "There is no file or folder with this name. Check the path and try again."
                    .to_owned(),
            });
        }
    }
    let mut seen: HashSet<PathBuf> = HashSet::new();
    let mut unique: Vec<PathBuf> = files
        .into_iter()
        .filter(|f| seen.insert(canonical(f)))
        .map(|f| std::path::absolute(&f).unwrap_or(f))
        .collect();
    unique.sort_by_key(|p| p.to_string_lossy().to_lowercase());
    out.files = unique;
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn touch(dir: &Path, name: &str) -> PathBuf {
        let path = dir.join(name);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("create folder");
        }
        std::fs::write(&path, b"x").expect("write");
        path
    }

    #[test]
    fn paths_from_stdin_skip_blank_lines_and_carriage_returns() {
        let text = "a.ARW\r\n\n  \nb c.ARW\n";
        let paths = read_paths(text.as_bytes()).expect("read");
        assert_eq!(paths, [PathBuf::from("a.ARW"), PathBuf::from("b c.ARW")]);
    }

    #[test]
    fn a_dash_is_replaced_by_the_paths_on_stdin_once() {
        let inputs = [
            PathBuf::from("first.ARW"),
            PathBuf::from("-"),
            PathBuf::from("-"),
        ];
        let out = with_stdin(&inputs, "x.ARW\ny.ARW\n".as_bytes()).expect("read");
        assert_eq!(
            out,
            [
                PathBuf::from("first.ARW"),
                PathBuf::from("x.ARW"),
                PathBuf::from("y.ARW")
            ]
        );
    }

    #[test]
    fn folders_are_walked_for_arw_files_in_any_case_and_hidden_files_are_left_out() {
        let dir = tempfile::tempdir().expect("tempdir");
        touch(dir.path(), "b.ARW");
        touch(dir.path(), "A.arw");
        touch(dir.path(), "c.jpg");
        touch(dir.path(), "._d.ARW");
        touch(dir.path(), "sub/e.ARW");
        let flat = expand(&[dir.path().to_path_buf()], false);
        let names = |e: &Expanded| -> Vec<String> {
            e.files
                .iter()
                .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
                .collect()
        };
        assert_eq!(names(&flat), ["A.arw", "b.ARW"]);
        assert_eq!(
            flat.ignored, 2,
            "c.jpg and ._d.ARW; a sub folder is not an entry that was passed over"
        );
        assert!(flat.problems.is_empty());
        let deep = expand(&[dir.path().to_path_buf()], true);
        assert_eq!(names(&deep), ["A.arw", "b.ARW", "e.ARW"]);
    }

    #[test]
    fn a_file_named_twice_is_one_photo_and_bad_names_are_problems() {
        let dir = tempfile::tempdir().expect("tempdir");
        let a = touch(dir.path(), "a.ARW");
        let notes = touch(dir.path(), "notes.txt");
        let missing = dir.path().join("gone.ARW");
        let out = expand(
            &[a.clone(), a.clone(), notes.clone(), missing.clone()],
            false,
        );
        assert_eq!(out.files, [a]);
        assert_eq!(out.problems.len(), 2);
        assert_eq!(out.problems[0].path, notes);
        assert_eq!(out.problems[0].kind, "not_arw");
        assert_eq!(out.problems[1].path, missing);
        assert_eq!(out.problems[1].kind, "not_found");
        assert!(out.problems[1].message.contains("Check the path"));
    }
}
