//! File names, destinations and safe writing (`docs/09-export.md` sections Options and Queue).
//!
//! * **Names.** The template (`{name}_{look}` by default) is expanded with `{name}` (the source
//!   file's name without extension), `{look}` (the look's slug), `{date}` (the capture date,
//!   `YYYY-MM-DD`), `{seq}` (the 1-based number of the photo in the job, at least three digits) and
//!   `{camera}`. Unknown `{words}` stay as typed, as in the UI preview. The result is made safe for a
//!   file name: path separators and characters Windows and exFAT reject become `_`, leading dots and
//!   trailing dots and spaces are dropped, the length is capped.
//! * **Destination.** `<source folder> export` next to the source folder, named as the person gave
//!   the path (symbolic links are not resolved), or `~/Pictures/Yellowmoon export` when that can't
//!   be written. An explicitly requested folder is never replaced by a fallback.
//! * **If the file exists.** Skip, Replace, or Keep both (" 2", " 3", ...). Keep both and Skip never
//!   overwrite, even when another process creates the file while we work.
//! * **Atomic write.** The file is written under a temporary name in the destination folder, flushed
//!   and fsynced, then renamed. A failure, a cancel or a panic deletes the temporary file, so a
//!   partial export never has its final name. Originals are never opened for writing; a path that
//!   resolves to the source file is refused.

use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::error::ExportError;
use crate::options::{DEFAULT_NAME_TEMPLATE, IfExists};

/// The values a file name template can use.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NameFields {
    /// The source file's name without extension (`DSC04204`).
    pub name: String,
    /// The look's slug (`neutral`): the part of the look id after its last `:`, as the export panel shows it.
    pub look: String,
    /// The capture date as `YYYY-MM-DD`; `None` expands to `undated`.
    pub date: Option<String>,
    /// The 1-based number of the photo in this export job.
    pub seq: u32,
    /// The camera as a [`slugify`]d body name (`sony-a1`), the same text the export panel previews.
    pub camera: String,
}

/// A lower-case word for a file name: every run of characters other than `a-z` and `0-9` becomes one
/// `-`, and none is left at either end (`Sony A1` gives `sony-a1`). The export panel does the same for
/// the `{camera}` token, so the file the core writes has the name the panel promised.
#[must_use]
pub fn slugify(text: &str) -> String {
    let mut out = String::new();
    let mut gap = false;
    for c in text.chars().flat_map(char::to_lowercase) {
        if c.is_ascii_lowercase() || c.is_ascii_digit() {
            if gap && !out.is_empty() {
                out.push('-');
            }
            gap = false;
            out.push(c);
        } else {
            gap = true;
        }
    }
    out
}

/// The longest file stem produced, in bytes. Leaves room for an extension, a " 99" suffix and the
/// 255-byte limit of APFS and HFS+ names.
pub const MAX_STEM_BYTES: usize = 200;

/// Expands `template` with `fields`. Unknown tokens are left as typed.
#[must_use]
pub fn expand_template(template: &str, fields: &NameFields) -> String {
    let mut out = String::with_capacity(template.len() + 16);
    let mut rest = template;
    while let Some(open) = rest.find('{') {
        out.push_str(&rest[..open]);
        let after = &rest[open + 1..];
        let Some(close) = after.find('}') else {
            out.push_str(&rest[open..]);
            return out;
        };
        let token = &after[..close];
        match token {
            "name" => out.push_str(&fields.name),
            "look" => out.push_str(&fields.look),
            "date" => out.push_str(fields.date.as_deref().unwrap_or("undated")),
            "seq" => out.push_str(&format!("{:03}", fields.seq)),
            "camera" => out.push_str(&fields.camera),
            _ => {
                out.push('{');
                out.push_str(token);
                out.push('}');
            }
        }
        rest = &after[close + 1..];
    }
    out.push_str(rest);
    out
}

/// Makes `text` safe as a file name stem (no extension).
#[must_use]
pub fn sanitize_stem(text: &str) -> String {
    let replaced: String = text
        .chars()
        .map(|c| match c {
            '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '_',
            c if c.is_control() => '_',
            c => c,
        })
        .collect();
    let trimmed = replaced
        .trim_start_matches(|c: char| c == '.' || c.is_whitespace())
        .trim_end_matches(|c: char| c == '.' || c.is_whitespace());
    let mut out = String::new();
    for c in trimmed.chars() {
        if out.len() + c.len_utf8() > MAX_STEM_BYTES {
            break;
        }
        out.push(c);
    }
    let out = out.trim_end_matches(|c: char| c == '.' || c.is_whitespace());
    if out.is_empty() {
        "export".to_owned()
    } else {
        out.to_owned()
    }
}

/// The file stem for a photo: the template (the default if empty) expanded and made safe.
#[must_use]
pub fn file_stem(template: &str, fields: &NameFields) -> String {
    let template = if template.trim().is_empty() {
        DEFAULT_NAME_TEMPLATE
    } else {
        template
    };
    sanitize_stem(&expand_template(template, fields))
}

/// `<source folder> export` next to the source folder: for `/Volumes/A1/Lisbon/DSC04204.ARW` that is
/// `/Volumes/A1/Lisbon export`. `None` when the source has no named parent folder with a parent of
/// its own (a file at the root of a volume).
///
/// The folder keeps the name the person knows it by. A symbolic link in the path is **not**
/// resolved: on a Mac `/var` is a link to `/private/var`, a photos folder is often a link to
/// another disk, and the Export panel, the command `ym render` is shown as and the files
/// themselves must all say `/var/…/Lisbon export` when the photo was opened as `/var/…/Lisbon`
/// (the session's `default_export_folder` is spelled the same way, `docs/01`). A relative path is
/// made absolute against the working directory. A folder that has no name as written, because the
/// path ends in `..` or has no folder part (`DSC04204.ARW` in the working directory), takes the
/// name of the folder it resolves to. The name is kept byte for byte, also when it is not UTF-8.
#[must_use]
pub fn default_destination(source_file: &Path) -> Option<PathBuf> {
    let folder = source_file.parent()?;
    let folder = if folder.as_os_str().is_empty() {
        std::env::current_dir().ok()?
    } else {
        std::path::absolute(folder).ok()?
    };
    let folder = if folder.file_name().is_some() {
        folder
    } else {
        fs::canonicalize(&folder).ok()?
    };
    let mut name = folder.file_name()?.to_os_string();
    name.push(" export");
    Some(folder.parent()?.join(name))
}

/// `~/Pictures/Yellowmoon export`, the fallback destination.
#[must_use]
pub fn fallback_destination(home: &Path) -> PathBuf {
    home.join("Pictures").join("Yellowmoon export")
}

/// The user's home folder from `$HOME`.
#[must_use]
pub fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .filter(|p| !p.as_os_str().is_empty())
}

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_name() -> String {
    format!(
        ".yellowmoon-{}-{}.tmp",
        std::process::id(),
        TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
    )
}

/// Creates `dir` if needed and checks that a file can be created in it.
pub fn ensure_writable_dir(dir: &Path) -> Result<(), ExportError> {
    fs::create_dir_all(dir).map_err(|e| ExportError::io(dir, e))?;
    let probe = dir.join(temp_name());
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&probe)
        .map_err(|e| ExportError::io(dir, e))?;
    drop(file);
    fs::remove_file(&probe).map_err(|e| ExportError::io(&probe, e))
}

/// Chooses and prepares the destination folder.
///
/// * `requested`: the user's folder. It is created if missing; if it can't be written the export
///   fails (no silent fallback).
/// * Otherwise the default next to the source folder, and if that can't be written
///   `~/Pictures/Yellowmoon export` under `home`.
pub fn resolve_destination_dir(
    requested: Option<&Path>,
    source_file: Option<&Path>,
    home: Option<&Path>,
) -> Result<PathBuf, ExportError> {
    if let Some(dir) = requested {
        ensure_writable_dir(dir)?;
        return Ok(dir.to_path_buf());
    }
    let mut last_error = None;
    if let Some(candidate) = source_file.and_then(default_destination) {
        match ensure_writable_dir(&candidate) {
            Ok(()) => return Ok(candidate),
            Err(e) => last_error = Some(e),
        }
    }
    if let Some(home) = home {
        let fallback = fallback_destination(home);
        ensure_writable_dir(&fallback)?;
        return Ok(fallback);
    }
    Err(last_error.unwrap_or_else(|| {
        ExportError::invalid("there is no folder to save to; choose one in the export options")
    }))
}

/// What happened to the output file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Placement {
    /// A new file was written at this path (for Keep both, possibly with a number added).
    Written(PathBuf),
    /// An existing file at this path was replaced.
    Replaced(PathBuf),
    /// A file already existed at this path and "Skip" was chosen; nothing was written.
    Skipped(PathBuf),
}

impl Placement {
    /// The path of the file (existing, for a skip).
    #[must_use]
    pub fn path(&self) -> &Path {
        match self {
            Self::Written(p) | Self::Replaced(p) | Self::Skipped(p) => p,
        }
    }
}

fn candidate(dir: &Path, stem: &str, ext: &str, n: u32) -> PathBuf {
    if n <= 1 {
        dir.join(format!("{stem}.{ext}"))
    } else {
        dir.join(format!("{stem} {n}.{ext}"))
    }
}

/// Whether `a` and `b` are the same file on disk.
fn same_file(a: &Path, b: &Path) -> bool {
    match (fs::canonicalize(a), fs::canonicalize(b)) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    }
}

fn check_not_original(path: &Path, source: Option<&Path>) -> Result<(), ExportError> {
    match source {
        Some(src) if same_file(path, src) => {
            Err(ExportError::WouldOverwriteOriginal(path.to_path_buf()))
        }
        _ => Ok(()),
    }
}

/// Decides before any work is done whether the export can proceed.
///
/// Returns `Some(Placement::Skipped(existing))` for "Skip" when the file exists, `None` otherwise.
/// Also refuses a target that resolves to the original.
pub fn plan_output(
    dir: &Path,
    stem: &str,
    ext: &str,
    if_exists: IfExists,
    source: Option<&Path>,
) -> Result<Option<Placement>, ExportError> {
    let target = candidate(dir, stem, ext, 1);
    if target.exists() {
        check_not_original(&target, source)?;
        if if_exists == IfExists::Skip {
            return Ok(Some(Placement::Skipped(target)));
        }
    }
    Ok(None)
}

/// A file being written under a temporary name in its destination folder.
///
/// Dropping it without calling [`PendingFile::commit`] deletes the temporary file.
#[derive(Debug)]
pub struct PendingFile {
    dir: PathBuf,
    temp: PathBuf,
    writer: Option<BufWriter<File>>,
    done: bool,
}

impl PendingFile {
    /// Creates the temporary file in `dir` (which must exist).
    pub fn create(dir: &Path) -> Result<Self, ExportError> {
        for _ in 0..16 {
            let temp = dir.join(temp_name());
            match OpenOptions::new().write(true).create_new(true).open(&temp) {
                Ok(file) => {
                    return Ok(Self {
                        dir: dir.to_path_buf(),
                        temp,
                        writer: Some(BufWriter::with_capacity(1 << 20, file)),
                        done: false,
                    });
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(e) => return Err(ExportError::io(dir, e)),
            }
        }
        Err(ExportError::io(
            dir,
            std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                "no free temporary file name",
            ),
        ))
    }

    /// The writer to hand to an encoder.
    pub fn writer(&mut self) -> &mut BufWriter<File> {
        // The writer is only taken by `commit`, which consumes `self`.
        self.writer
            .as_mut()
            .expect("the writer exists until commit")
    }

    /// The temporary path (for error messages).
    #[must_use]
    pub fn temp_path(&self) -> &Path {
        &self.temp
    }

    /// Flushes and fsyncs the temporary file, then moves it to its final name according to
    /// `if_exists`. The size of the file is returned with the placement.
    pub fn commit(
        mut self,
        stem: &str,
        ext: &str,
        if_exists: IfExists,
        source: Option<&Path>,
    ) -> Result<(Placement, u64), ExportError> {
        let writer = self
            .writer
            .take()
            .ok_or_else(|| ExportError::invalid("the file was already committed"))?;
        let file = writer
            .into_inner()
            .map_err(|e| ExportError::io(&self.temp, e.into_error()))?;
        file.sync_all()
            .map_err(|e| ExportError::io(&self.temp, e))?;
        let size = file
            .metadata()
            .map_err(|e| ExportError::io(&self.temp, e))?
            .len();
        drop(file);

        let first = candidate(&self.dir, stem, ext, 1);
        let placement = match if_exists {
            IfExists::Replace => {
                if first.is_dir() {
                    return Err(ExportError::CannotReplace(first));
                }
                let existed = first.exists();
                if existed {
                    check_not_original(&first, source)?;
                }
                fs::rename(&self.temp, &first).map_err(|e| ExportError::io(&first, e))?;
                if existed {
                    Placement::Replaced(first)
                } else {
                    Placement::Written(first)
                }
            }
            IfExists::Skip => {
                if first.exists() {
                    check_not_original(&first, source)?;
                    return Ok((Placement::Skipped(first), 0));
                }
                match rename_noclobber(&self.temp, &first) {
                    Ok(()) => Placement::Written(first),
                    Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                        return Ok((Placement::Skipped(first), 0));
                    }
                    Err(e) => return Err(ExportError::io(&first, e)),
                }
            }
            IfExists::KeepBoth => {
                let mut n = 1;
                loop {
                    let target = candidate(&self.dir, stem, ext, n);
                    match rename_noclobber(&self.temp, &target) {
                        Ok(()) => break Placement::Written(target),
                        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists && n < 10_000 => {
                            n += 1
                        }
                        Err(e) => return Err(ExportError::io(&target, e)),
                    }
                }
            }
        };
        self.done = true;
        sync_dir(&self.dir);
        Ok((placement, size))
    }
}

impl Drop for PendingFile {
    fn drop(&mut self) {
        if !self.done {
            // Close the file first (Windows would refuse to delete an open file), then remove it.
            self.writer = None;
            let _ = fs::remove_file(&self.temp);
        }
    }
}

/// Moves `from` to `to` without replacing an existing `to`, atomically where the file system
/// allows: a hard link then unlink, or, on file systems without hard links (FAT, exFAT), a check
/// and a rename.
fn rename_noclobber(from: &Path, to: &Path) -> std::io::Result<()> {
    match fs::hard_link(from, to) {
        Ok(()) => {
            let _ = fs::remove_file(from);
            Ok(())
        }
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => Err(e),
        Err(_) => {
            if to.exists() {
                return Err(std::io::Error::from(std::io::ErrorKind::AlreadyExists));
            }
            fs::rename(from, to)
        }
    }
}

#[cfg(unix)]
fn sync_dir(dir: &Path) {
    if let Ok(d) = File::open(dir) {
        let _ = d.sync_all();
    }
}

#[cfg(not(unix))]
fn sync_dir(_dir: &Path) {}

/// Runs `body` against a temporary file in `dir`, then commits it. The temporary file is removed if
/// `body` fails.
pub fn write_atomically<F>(
    dir: &Path,
    stem: &str,
    ext: &str,
    if_exists: IfExists,
    source: Option<&Path>,
    body: F,
) -> Result<(Placement, u64), ExportError>
where
    F: FnOnce(&mut BufWriter<File>) -> Result<(), ExportError>,
{
    let mut pending = PendingFile::create(dir)?;
    body(pending.writer())?;
    pending
        .writer()
        .flush()
        .map_err(|e| ExportError::io(pending.temp_path(), e))?;
    pending.commit(stem, ext, if_exists, source)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fields() -> NameFields {
        NameFields {
            name: "DSC04204".into(),
            look: "neutral".into(),
            date: Some("2026-10-04".into()),
            seq: 7,
            camera: "ILCE-1".into(),
        }
    }

    #[test]
    fn templates_expand_every_token() {
        let f = fields();
        assert_eq!(expand_template("{name}_{look}", &f), "DSC04204_neutral");
        assert_eq!(
            expand_template("{date} {camera} {seq} {name}", &f),
            "2026-10-04 ILCE-1 007 DSC04204"
        );
        assert_eq!(expand_template("{nope}-{name}", &f), "{nope}-DSC04204");
        assert_eq!(expand_template("a{name", &f), "a{name");
        assert_eq!(
            expand_template(
                "{seq}",
                &NameFields {
                    seq: 1234,
                    ..f.clone()
                }
            ),
            "1234"
        );
        assert_eq!(
            expand_template("{date}", &NameFields { date: None, ..f }),
            "undated"
        );
    }

    #[test]
    fn slugs_match_the_export_panel() {
        // `slugify` in src/export/template.ts.
        assert_eq!(slugify("Sony A1"), "sony-a1");
        assert_eq!(slugify("ILCE-7RM4"), "ilce-7rm4");
        assert_eq!(slugify("  Sony  \u{3b1}7 IV!! "), "sony-7-iv");
        assert_eq!(slugify("a__b--c"), "a-b-c");
        assert_eq!(slugify("???"), "");
        assert_eq!(slugify(""), "");
        assert_eq!(slugify("ILCE-1"), "ilce-1");
    }

    #[test]
    fn stems_are_made_safe() {
        assert_eq!(
            sanitize_stem("a/b\\c:d*e?f\"g<h>i|j"),
            "a_b_c_d_e_f_g_h_i_j"
        );
        assert_eq!(sanitize_stem("..hidden"), "hidden");
        assert_eq!(sanitize_stem("  spaced out.  "), "spaced out");
        assert_eq!(sanitize_stem("tab\there"), "tab_here");
        assert_eq!(sanitize_stem(""), "export");
        assert_eq!(sanitize_stem("..."), "export");
        let long = "é".repeat(300);
        let s = sanitize_stem(&long);
        assert!(s.len() <= MAX_STEM_BYTES);
        assert!(s.chars().all(|c| c == 'é'));
        // A template with slashes can't escape the folder.
        let escaped = file_stem("../../{name}", &fields());
        assert_eq!(escaped, "_.._DSC04204");
        assert!(!escaped.contains('/'));
    }

    #[test]
    fn empty_template_uses_the_default() {
        assert_eq!(file_stem("", &fields()), "DSC04204_neutral");
        assert_eq!(file_stem("   ", &fields()), "DSC04204_neutral");
    }

    #[test]
    fn default_destination_is_next_to_the_source_folder() {
        assert_eq!(
            default_destination(Path::new("/Volumes/A1/Lisbon/DSC04204.ARW")),
            Some(PathBuf::from("/Volumes/A1/Lisbon export"))
        );
        assert_eq!(default_destination(Path::new("/DSC04204.ARW")), None);
        assert_eq!(
            fallback_destination(Path::new("/Users/me")),
            PathBuf::from("/Users/me/Pictures/Yellowmoon export")
        );
    }

    #[test]
    fn destination_resolution_and_fallback() {
        let tmp = tempfile::tempdir().unwrap();
        let src_dir = tmp.path().join("Lisbon");
        fs::create_dir_all(&src_dir).unwrap();
        let src = src_dir.join("DSC1.ARW");
        fs::write(&src, b"raw").unwrap();
        let home = tmp.path().join("home");
        fs::create_dir_all(&home).unwrap();

        let dir = resolve_destination_dir(None, Some(&src), Some(&home)).unwrap();
        assert_eq!(
            dir,
            fs::canonicalize(tmp.path()).unwrap().join("Lisbon export")
        );
        assert!(dir.is_dir());

        // Explicit folders are created and used as given.
        let chosen = tmp.path().join("chosen/deeper");
        assert_eq!(
            resolve_destination_dir(Some(&chosen), Some(&src), None).unwrap(),
            chosen
        );
        assert!(chosen.is_dir());

        // No source and no home: an error that says what to do.
        let err = resolve_destination_dir(None, None, None).unwrap_err();
        assert!(err.to_string().contains("choose one"), "{err}");

        // A source whose "<folder> export" can't be created falls back to ~/Pictures/Yellowmoon export.
        // Make the sibling name a file so create_dir_all fails.
        let blocked_src_dir = tmp.path().join("Blocked");
        fs::create_dir_all(&blocked_src_dir).unwrap();
        fs::write(tmp.path().join("Blocked export"), b"i am a file").unwrap();
        let blocked_src = blocked_src_dir.join("DSC2.ARW");
        fs::write(&blocked_src, b"raw").unwrap();
        let dir = resolve_destination_dir(None, Some(&blocked_src), Some(&home)).unwrap();
        assert_eq!(dir, home.join("Pictures/Yellowmoon export"));
        // An explicit unwritable folder does NOT fall back.
        let explicit = tmp.path().join("Blocked export");
        assert!(resolve_destination_dir(Some(&explicit), Some(&blocked_src), Some(&home)).is_err());
    }

    fn write_file(
        dir: &Path,
        stem: &str,
        ext: &str,
        mode: IfExists,
        content: &[u8],
        source: Option<&Path>,
    ) -> Result<(Placement, u64), ExportError> {
        write_atomically(dir, stem, ext, mode, source, |w| {
            w.write_all(content).map_err(|e| ExportError::io(dir, e))
        })
    }

    #[test]
    fn keep_both_numbers_the_files_and_never_overwrites() {
        let tmp = tempfile::tempdir().unwrap();
        let d = tmp.path();
        let a = write_file(d, "photo", "jpg", IfExists::KeepBoth, b"one", None).unwrap();
        let b = write_file(d, "photo", "jpg", IfExists::KeepBoth, b"two", None).unwrap();
        let c = write_file(d, "photo", "jpg", IfExists::KeepBoth, b"three", None).unwrap();
        assert_eq!(a.0, Placement::Written(d.join("photo.jpg")));
        assert_eq!(b.0, Placement::Written(d.join("photo 2.jpg")));
        assert_eq!(c.0, Placement::Written(d.join("photo 3.jpg")));
        assert_eq!(fs::read(d.join("photo.jpg")).unwrap(), b"one");
        assert_eq!(fs::read(d.join("photo 2.jpg")).unwrap(), b"two");
        assert_eq!(fs::read(d.join("photo 3.jpg")).unwrap(), b"three");
        assert_eq!(c.1, 5);
        assert_eq!(
            fs::read_dir(d).unwrap().count(),
            3,
            "no temporary files left"
        );
    }

    #[test]
    fn replace_swaps_the_file_atomically() {
        let tmp = tempfile::tempdir().unwrap();
        let d = tmp.path();
        let first = write_file(d, "photo", "png", IfExists::Replace, b"old", None).unwrap();
        assert_eq!(first.0, Placement::Written(d.join("photo.png")));
        let second = write_file(d, "photo", "png", IfExists::Replace, b"new", None).unwrap();
        assert_eq!(second.0, Placement::Replaced(d.join("photo.png")));
        assert_eq!(fs::read(d.join("photo.png")).unwrap(), b"new");
        assert_eq!(fs::read_dir(d).unwrap().count(), 1);
        // A directory in the way can't be replaced.
        fs::create_dir(d.join("blocked.png")).unwrap();
        let err = write_file(d, "blocked", "png", IfExists::Replace, b"x", None).unwrap_err();
        assert!(matches!(err, ExportError::CannotReplace(_)));
        assert_eq!(
            fs::read_dir(d).unwrap().count(),
            2,
            "temporary file removed after the failure"
        );
    }

    #[test]
    fn skip_leaves_the_existing_file_alone() {
        let tmp = tempfile::tempdir().unwrap();
        let d = tmp.path();
        fs::write(d.join("photo.tif"), b"existing").unwrap();
        let planned = plan_output(d, "photo", "tif", IfExists::Skip, None).unwrap();
        assert_eq!(planned, Some(Placement::Skipped(d.join("photo.tif"))));
        assert_eq!(
            plan_output(d, "photo", "tif", IfExists::KeepBoth, None).unwrap(),
            None
        );
        assert_eq!(
            plan_output(d, "other", "tif", IfExists::Skip, None).unwrap(),
            None
        );
        // Even if the file appears while we work, Skip does not overwrite.
        let r = write_file(d, "photo", "tif", IfExists::Skip, b"new", None).unwrap();
        assert_eq!(r.0, Placement::Skipped(d.join("photo.tif")));
        assert_eq!(fs::read(d.join("photo.tif")).unwrap(), b"existing");
        assert_eq!(fs::read_dir(d).unwrap().count(), 1);
    }

    #[test]
    fn failures_and_drops_delete_the_partial_file() {
        let tmp = tempfile::tempdir().unwrap();
        let d = tmp.path();
        let err = write_atomically(d, "x", "jpg", IfExists::KeepBoth, None, |w| {
            w.write_all(b"partial").unwrap();
            Err(ExportError::Cancelled)
        })
        .unwrap_err();
        assert!(matches!(err, ExportError::Cancelled));
        assert_eq!(fs::read_dir(d).unwrap().count(), 0);
        // A pending file that is dropped (a panic, an early return) cleans up too.
        {
            let mut p = PendingFile::create(d).unwrap();
            p.writer().write_all(b"abc").unwrap();
            assert!(p.temp_path().exists());
        }
        assert_eq!(fs::read_dir(d).unwrap().count(), 0);
    }

    #[test]
    fn the_original_is_never_overwritten() {
        let tmp = tempfile::tempdir().unwrap();
        let d = tmp.path();
        let src = d.join("DSC1.dng");
        fs::write(&src, b"original").unwrap();
        let err = write_file(d, "DSC1", "dng", IfExists::Replace, b"new", Some(&src)).unwrap_err();
        assert!(matches!(err, ExportError::WouldOverwriteOriginal(_)));
        assert_eq!(fs::read(&src).unwrap(), b"original");
        assert!(matches!(
            plan_output(d, "DSC1", "dng", IfExists::Replace, Some(&src)),
            Err(ExportError::WouldOverwriteOriginal(_))
        ));
        // Keep both is fine: it writes "DSC1 2.dng".
        let r = write_file(d, "DSC1", "dng", IfExists::KeepBoth, b"new", Some(&src)).unwrap();
        assert_eq!(r.0, Placement::Written(d.join("DSC1 2.dng")));
        assert_eq!(fs::read(&src).unwrap(), b"original");
    }

    #[test]
    fn concurrent_keep_both_writers_never_collide() {
        let tmp = tempfile::tempdir().unwrap();
        let d = tmp.path().to_path_buf();
        let handles: Vec<_> = (0..8)
            .map(|i| {
                let d = d.clone();
                std::thread::spawn(move || {
                    write_file(
                        &d,
                        "same",
                        "jpg",
                        IfExists::KeepBoth,
                        format!("file {i}").as_bytes(),
                        None,
                    )
                    .unwrap()
                    .0
                })
            })
            .collect();
        let mut paths: Vec<PathBuf> = handles
            .into_iter()
            .map(|h| h.join().unwrap().path().to_path_buf())
            .collect();
        paths.sort();
        paths.dedup();
        assert_eq!(paths.len(), 8, "eight different files");
        assert_eq!(fs::read_dir(&d).unwrap().count(), 8);
    }

    proptest::proptest! {
        #![proptest_config(proptest::prelude::ProptestConfig::with_cases(512))]

        /// Whatever the user types, the stem is a single, non-hidden, non-empty path component.
        #[test]
        fn stems_are_always_a_safe_single_component(text in ".*") {
            let stem = sanitize_stem(&text);
            proptest::prop_assert!(!stem.is_empty());
            proptest::prop_assert!(stem.len() <= MAX_STEM_BYTES);
            proptest::prop_assert!(!stem.starts_with('.'));
            proptest::prop_assert!(!stem.ends_with('.') && !stem.ends_with(' '));
            proptest::prop_assert!(!stem.chars().any(|c| matches!(c, '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|') || c.is_control()));
            // Sanitising twice changes nothing.
            proptest::prop_assert_eq!(sanitize_stem(&stem), stem);
        }

        /// Template expansion never panics and keeps the text around the tokens.
        #[test]
        fn templates_never_panic(template in ".*", seq in 0_u32..100_000) {
            let f = NameFields { seq, ..fields() };
            let out = expand_template(&template, &f);
            let _ = file_stem(&template, &f);
            let mut literal = template.clone();
            for token in ["{name}", "{look}", "{date}", "{seq}", "{camera}"] {
                literal = literal.replace(token, "");
            }
            proptest::prop_assert!(out.len() >= literal.len());
        }
    }
}
