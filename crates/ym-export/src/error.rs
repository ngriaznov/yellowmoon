//! Errors of the export crate. Messages are user-facing: they say what went wrong and what to do next
//! (`AGENTS.md`, UI copy).

use std::path::{Path, PathBuf};

/// Everything that can go wrong while exporting a photo.
#[derive(Debug, thiserror::Error)]
pub enum ExportError {
    /// The request itself is wrong (a programming error or an out-of-range option).
    #[error("This export can't be written: {0}.")]
    InvalidInput(String),
    /// The format, size or colour space can't be written by this crate.
    #[error("{0}")]
    Unsupported(String),
    /// A file or folder operation failed. `path` is always the destination folder (the temporary
    /// file's name is never shown: it is hidden and means nothing to the person).
    #[error("{}", io_text(path, source))]
    Io {
        /// The destination folder.
        path: PathBuf,
        /// The operating system error. Its text is for the log and `--json` ([`ExportError::detail`]),
        /// never for the sentence.
        #[source]
        source: std::io::Error,
    },
    /// An encoder could not write its output (a full disk, a folder that went away). The encoder
    /// reports it without a path; the destination folder is filled in by the writer that owns the file.
    #[error("{}", write_text(format, folder.as_deref(), source))]
    Write {
        /// What was being written ("JPEG", "TIFF", "PNG", "DNG").
        format: &'static str,
        /// The destination folder, when the writer that owns the file knows it.
        folder: Option<PathBuf>,
        /// The operating system error (see [`ExportError::Io`]).
        #[source]
        source: std::io::Error,
    },
    /// The job was cancelled; the partial file has been deleted.
    #[error("The export was cancelled and the partial file was removed.")]
    Cancelled,
    /// An image or metadata encoder failed.
    #[error("Encoding the {format} failed: {reason}.")]
    Encode {
        /// What was being written ("JPEG", "TIFF", "PNG", "DNG", "XMP", "EXIF", "ICC profile").
        format: &'static str,
        /// The encoder's message.
        reason: String,
    },
    /// The output path would overwrite one of the photo's original files.
    #[error(
        "The export would overwrite the original file {}. Pick another folder or file name.",
        .0.display()
    )]
    WouldOverwriteOriginal(PathBuf),
    /// The destination is a directory, or otherwise can't be replaced.
    #[error("{} can't be replaced. Remove it or pick another name.", .0.display())]
    CannotReplace(PathBuf),
}

impl ExportError {
    /// An [`ExportError::Io`] for `path`.
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }

    /// An [`ExportError::Write`]: the encoder of `format` could not write to its output.
    pub fn write(format: &'static str, source: std::io::Error) -> Self {
        Self::Write {
            format,
            folder: None,
            source,
        }
    }

    /// Names the file format in an encoder error that was raised under another name. A DNG is
    /// written by the TIFF encoder, whose errors say "TIFF"; the person asked for a DNG.
    pub(crate) fn labelled(self, format: &'static str) -> Self {
        match self {
            Self::Encode {
                format: "TIFF",
                reason,
            } => Self::Encode { format, reason },
            Self::Write {
                format: "TIFF",
                folder,
                source,
            } => Self::Write {
                format,
                folder,
                source,
            },
            other => other,
        }
    }

    /// An [`ExportError::Encode`] for `format`.
    pub fn encode(format: &'static str, reason: impl std::fmt::Display) -> Self {
        Self::Encode {
            format,
            reason: reason.to_string(),
        }
    }

    /// An [`ExportError::InvalidInput`].
    pub fn invalid(reason: impl Into<String>) -> Self {
        Self::InvalidInput(reason.into())
    }
}

/// Why the destination, not the photo, is the problem. A failure of this kind will repeat for
/// every photo of the job (`docs/26-hardening.md` section 2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DestinationProblem {
    /// The disk (or the quota) is full.
    DiskFull,
    /// The folder is on a read-only volume.
    ReadOnly,
    /// The folder or its drive is not there.
    Gone,
    /// The app is not allowed to save there.
    Permission,
}

impl DestinationProblem {
    /// Classifies an I/O error, or `None` when it says nothing about the destination as a whole.
    #[must_use]
    pub fn of_io(e: &std::io::Error) -> Option<Self> {
        if e.raw_os_error().is_none()
            && let Some(code) = code_in_text(e)
        {
            return Self::of_io(&std::io::Error::from_raw_os_error(code));
        }
        use std::io::ErrorKind as K;
        match e.kind() {
            K::StorageFull | K::QuotaExceeded => return Some(Self::DiskFull),
            K::ReadOnlyFilesystem => return Some(Self::ReadOnly),
            K::NotFound => return Some(Self::Gone),
            K::PermissionDenied => return Some(Self::Permission),
            _ => {}
        }
        // The codes of `errno` that the kinds do not cover on every platform. 69 is `EDQUOT` on
        // macOS only (on Linux it is an unrelated code, and `QuotaExceeded` above covers 122).
        match e.raw_os_error()? {
            28 => Some(Self::DiskFull),
            #[cfg(target_vendor = "apple")]
            69 => Some(Self::DiskFull),
            30 => Some(Self::ReadOnly),
            // `ENOENT`, `ENXIO` (device not configured) and `ENODEV`: a drive that was pulled.
            2 | 6 | 19 => Some(Self::Gone),
            // `EACCES` and `EPERM` (the second is what a macOS privacy block gives).
            13 | 1 => Some(Self::Permission),
            _ => None,
        }
    }

    /// The sentence for the app. `folder` is the destination folder; only its last component is
    /// said. Without a folder the sentence says "The folder".
    #[must_use]
    pub fn app_message(self, folder: Option<&Path>) -> String {
        let name = folder.map_or_else(|| "The folder".to_owned(), folder_name);
        match self {
            Self::DiskFull => "The disk is full, so the export stopped. Free some space, then export the photos that are left.".to_owned(),
            Self::ReadOnly => format!("{name} is read-only. Choose another folder."),
            Self::Gone => format!("Yellowmoon can't reach {name}. If it is on a drive, check that the drive is connected, then export again."),
            Self::Permission => format!("Yellowmoon isn't allowed to save to {name}. Choose another folder, or allow access in System Settings, under Privacy & Security."),
        }
    }

    /// The sentence for `ym` (stderr), which names the folder by its full path.
    #[must_use]
    pub fn cli_message(self, folder: &Path) -> String {
        let path = folder.display();
        match self {
            Self::DiskFull => format!(
                "Stopped: the disk holding {path} is full. Free some space, then run the command again with --skip-existing."
            ),
            Self::ReadOnly => format!("{path} is read-only. Choose another folder with --out."),
            Self::Gone => format!(
                "{path} is not there any more. Check the path or the drive, then run the command again."
            ),
            Self::Permission => format!(
                "Yellowmoon isn't allowed to write to {path}. Choose another folder with --out."
            ),
        }
    }
}

/// The `errno` of an error that an encoder turned into text. The PNG encoder's stream writer
/// hands an I/O error on as `io::Error::other(text)`, which keeps "(os error 28)" and loses the
/// code. A wrapped error whose kind is not `Other` is left alone.
fn code_in_text(e: &std::io::Error) -> Option<i32> {
    if e.kind() != std::io::ErrorKind::Other {
        return None;
    }
    let text = e.to_string();
    let rest = text.split("(os error ").nth(1)?;
    rest.split(')').next()?.trim().parse().ok()
}

/// The last component of `folder`, or the whole path for a folder that has none (`/`).
fn folder_name(folder: &Path) -> String {
    folder.file_name().map_or_else(
        || folder.display().to_string(),
        |n| n.to_string_lossy().into_owned(),
    )
}

const GENERIC_ADVICE: &str = "Check that the folder is still there and that you can save to it.";

fn io_text(path: &Path, source: &std::io::Error) -> String {
    match DestinationProblem::of_io(source) {
        Some(problem) => problem.app_message(Some(path)),
        None => format!("Couldn't write to {}. {GENERIC_ADVICE}", path.display()),
    }
}

fn write_text(format: &str, folder: Option<&Path>, source: &std::io::Error) -> String {
    match DestinationProblem::of_io(source) {
        Some(problem) => problem.app_message(folder),
        None => format!("Couldn't write the {format}. {GENERIC_ADVICE}"),
    }
}

impl ExportError {
    /// Whether the failure is about the destination as a whole (a full disk, a read-only volume, a
    /// folder that is gone, no permission) rather than this photo: the next photo will meet it too.
    /// Looks at the I/O error of [`ExportError::Io`] and [`ExportError::Write`].
    #[must_use]
    pub fn destination_problem(&self) -> Option<DestinationProblem> {
        match self {
            Self::Io { source, .. } | Self::Write { source, .. } => {
                DestinationProblem::of_io(source)
            }
            _ => None,
        }
    }

    /// The destination folder of an I/O failure, when it is known.
    #[must_use]
    pub fn folder(&self) -> Option<&Path> {
        match self {
            Self::Io { path, .. } => Some(path),
            Self::Write { folder, .. } => folder.as_deref(),
            _ => None,
        }
    }

    /// The operating system's own text of an I/O failure ("No space left on device (os error 28)"),
    /// for the log and `--json` (`error.detail`). Never part of a sentence.
    #[must_use]
    pub fn detail(&self) -> Option<String> {
        match self {
            Self::Io { source, .. } | Self::Write { source, .. } => Some(source.to_string()),
            _ => None,
        }
    }

    /// Records the destination folder in an encoder error that does not know it (the writer that owns the file calls this).
    #[must_use]
    pub fn with_folder(self, dir: &Path) -> Self {
        match self {
            Self::Write {
                format,
                folder: None,
                source,
            } => Self::Write {
                format,
                folder: Some(dir.to_path_buf()),
                source,
            },
            other => other,
        }
    }
}

impl From<jpeg_encoder::EncodingError> for ExportError {
    fn from(e: jpeg_encoder::EncodingError) -> Self {
        match e {
            jpeg_encoder::EncodingError::IoError(io) => Self::write("JPEG", io),
            other => Self::encode("JPEG", other),
        }
    }
}

impl From<png::EncodingError> for ExportError {
    fn from(e: png::EncodingError) -> Self {
        match e {
            png::EncodingError::IoError(io) => Self::write("PNG", io),
            other => Self::encode("PNG", other),
        }
    }
}

impl From<tiff::TiffError> for ExportError {
    fn from(e: tiff::TiffError) -> Self {
        match e {
            tiff::TiffError::IoError(io) => Self::write("TIFF", io),
            other => Self::encode("TIFF", other),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;

    fn full() -> io::Error {
        io::Error::from_raw_os_error(28)
    }

    #[test]
    fn io_failures_of_the_encoders_say_what_to_do_and_hide_the_os_text() {
        for (e, _format) in [
            (
                ExportError::from(png::EncodingError::IoError(full())),
                "PNG",
            ),
            (ExportError::from(tiff::TiffError::IoError(full())), "TIFF"),
            (
                ExportError::from(jpeg_encoder::EncodingError::IoError(full())),
                "JPEG",
            ),
        ] {
            let text = e.to_string();
            assert!(text.starts_with("The disk is full"), "{text}");
            assert!(text.contains("Free some space"), "{text}");
            assert!(!text.contains("os error"), "{text}");
            assert!(
                e.detail().is_some_and(|d| d.contains("No space left")),
                "{e:?}"
            );
        }
    }

    #[test]
    fn other_io_failures_get_the_generic_sentence() {
        let other = io::Error::from(io::ErrorKind::InvalidInput);
        let e = ExportError::write("PNG", other);
        assert_eq!(
            e.to_string(),
            "Couldn't write the PNG. Check that the folder is still there and that you can save to it."
        );
        assert_eq!(e.destination_problem(), None);
        let e = ExportError::io("/pics/out", io::Error::from(io::ErrorKind::InvalidInput));
        assert_eq!(
            e.to_string(),
            "Couldn't write to /pics/out. Check that the folder is still there and that you can save to it."
        );
        assert!(e.detail().is_some());
    }

    #[test]
    fn each_error_is_classified() {
        use DestinationProblem::*;
        let kinds = [
            (io::ErrorKind::StorageFull, DiskFull),
            (io::ErrorKind::QuotaExceeded, DiskFull),
            (io::ErrorKind::ReadOnlyFilesystem, ReadOnly),
            (io::ErrorKind::NotFound, Gone),
            (io::ErrorKind::PermissionDenied, Permission),
        ];
        for (kind, want) in kinds {
            assert_eq!(
                DestinationProblem::of_io(&io::Error::from(kind)),
                Some(want)
            );
        }
        // Codes that are the same on Linux and macOS.
        for (code, want) in [
            (28, DiskFull),
            (30, ReadOnly),
            (2, Gone),
            (6, Gone),
            (19, Gone),
            (13, Permission),
            (1, Permission),
        ] {
            assert_eq!(
                DestinationProblem::of_io(&io::Error::from_raw_os_error(code)),
                Some(want),
                "errno {code}"
            );
        }
        // `EDQUOT`: 69 on macOS, 122 on Linux (where 69 means something else).
        #[cfg(target_vendor = "apple")]
        assert_eq!(
            DestinationProblem::of_io(&io::Error::from_raw_os_error(69)),
            Some(DiskFull)
        );
        #[cfg(target_os = "linux")]
        assert_eq!(
            DestinationProblem::of_io(&io::Error::from_raw_os_error(122)),
            Some(DiskFull)
        );
        // A bad argument, a file in the way and a full path are not the destination's fault.
        for code in [22, 20, 17, 36] {
            assert_eq!(
                DestinationProblem::of_io(&io::Error::from_raw_os_error(code)),
                None,
                "errno {code}"
            );
        }
        assert_eq!(ExportError::encode("PNG", "x").destination_problem(), None);
    }

    #[test]
    fn the_sentences_name_the_folder() {
        use DestinationProblem::*;
        let folder = Path::new("/Volumes/Photos/Lisbon");
        assert_eq!(
            DiskFull.app_message(Some(folder)),
            "The disk is full, so the export stopped. Free some space, then export the photos that are left."
        );
        assert_eq!(
            ReadOnly.app_message(Some(folder)),
            "Lisbon is read-only. Choose another folder."
        );
        assert_eq!(
            Gone.app_message(Some(folder)),
            "Yellowmoon can't reach Lisbon. If it is on a drive, check that the drive is connected, then export again."
        );
        assert_eq!(
            Permission.app_message(Some(folder)),
            "Yellowmoon isn't allowed to save to Lisbon. Choose another folder, or allow access in System Settings, under Privacy & Security."
        );
        assert_eq!(
            ReadOnly.app_message(None),
            "The folder is read-only. Choose another folder."
        );
        assert_eq!(
            DiskFull.cli_message(folder),
            "Stopped: the disk holding /Volumes/Photos/Lisbon is full. Free some space, then run the command again with --skip-existing."
        );
        assert_eq!(
            ReadOnly.cli_message(folder),
            "/Volumes/Photos/Lisbon is read-only. Choose another folder with --out."
        );
        assert_eq!(
            Gone.cli_message(folder),
            "/Volumes/Photos/Lisbon is not there any more. Check the path or the drive, then run the command again."
        );
        assert_eq!(
            Permission.cli_message(folder),
            "Yellowmoon isn't allowed to write to /Volumes/Photos/Lisbon. Choose another folder with --out."
        );
        // A folder with no last component is named whole.
        assert_eq!(
            ReadOnly.app_message(Some(Path::new("/"))),
            "/ is read-only. Choose another folder."
        );
    }

    #[test]
    fn an_io_error_names_its_folder_and_never_the_os_text() {
        let e = ExportError::io("/pics/Lisbon", io::Error::from_raw_os_error(30));
        assert_eq!(e.to_string(), "Lisbon is read-only. Choose another folder.");
        assert_eq!(e.folder(), Some(Path::new("/pics/Lisbon")));
        let w = ExportError::write("TIFF", io::Error::from_raw_os_error(13))
            .with_folder(Path::new("/pics/Rome"));
        assert!(w.to_string().contains("Rome"), "{w}");
        assert_eq!(
            w.destination_problem(),
            Some(DestinationProblem::Permission)
        );
    }

    #[test]
    fn relabelling_only_touches_the_tiff_name() {
        let tiff_io = ExportError::from(tiff::TiffError::IoError(io::Error::from(
            io::ErrorKind::InvalidInput,
        )))
        .labelled("DNG");
        assert!(tiff_io.to_string().starts_with("Couldn't write the DNG"));
        let kept = ExportError::from(tiff::TiffError::IoError(full()))
            .with_folder(Path::new("/x"))
            .labelled("DNG");
        assert_eq!(kept.folder(), Some(Path::new("/x")));
        let tiff_enc = ExportError::encode("TIFF", "bad").labelled("DNG");
        assert_eq!(tiff_enc.to_string(), "Encoding the DNG failed: bad.");
        let other = ExportError::encode("XMP", "bad").labelled("DNG");
        assert_eq!(other.to_string(), "Encoding the XMP failed: bad.");
    }
}
