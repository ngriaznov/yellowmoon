//! Errors of the export crate. Messages are user-facing: they say what went wrong and what to do next
//! (`AGENTS.md`, UI copy).

use std::path::PathBuf;

/// Everything that can go wrong while exporting a photo.
#[derive(Debug, thiserror::Error)]
pub enum ExportError {
    /// The request itself is wrong (a programming error or an out-of-range option).
    #[error("This export can't be written: {0}.")]
    InvalidInput(String),
    /// The format, size or colour space can't be written by this crate.
    #[error("{0}")]
    Unsupported(String),
    /// A file or folder operation failed.
    #[error(
        "Couldn't write to {}: {source}. Check that the folder exists, is writable and has free space.",
        path.display()
    )]
    Io {
        /// The file or folder involved.
        path: PathBuf,
        /// The operating system error.
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

impl From<jpeg_encoder::EncodingError> for ExportError {
    fn from(e: jpeg_encoder::EncodingError) -> Self {
        Self::encode("JPEG", e)
    }
}

impl From<png::EncodingError> for ExportError {
    fn from(e: png::EncodingError) -> Self {
        match e {
            png::EncodingError::IoError(io) => Self::encode("PNG", io),
            other => Self::encode("PNG", other),
        }
    }
}

impl From<tiff::TiffError> for ExportError {
    fn from(e: tiff::TiffError) -> Self {
        Self::encode("TIFF", e)
    }
}
