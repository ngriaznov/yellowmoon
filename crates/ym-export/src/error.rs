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
    /// An encoder could not write its output (a full disk, a folder that went away). The encoder
    /// reports it without a path; the advice is the same as for [`ExportError::Io`].
    #[error(
        "Couldn't write the {format}: {source}. Check that the folder exists, is writable and has free space."
    )]
    Write {
        /// What was being written ("JPEG", "TIFF", "PNG", "DNG").
        format: &'static str,
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

    /// An [`ExportError::Write`]: the encoder of `format` could not write to its output.
    pub fn write(format: &'static str, source: std::io::Error) -> Self {
        Self::Write { format, source }
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
                source,
            } => Self::Write { format, source },
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
    fn io_failures_of_the_encoders_carry_the_next_step() {
        for (e, format) in [
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
            assert!(text.contains(format), "{text}");
            assert!(text.contains("No space left on device"), "{text}");
            assert!(text.contains("has free space"), "{text}");
        }
    }

    #[test]
    fn relabelling_only_touches_the_tiff_name() {
        let tiff_io = ExportError::from(tiff::TiffError::IoError(full())).labelled("DNG");
        assert!(tiff_io.to_string().starts_with("Couldn't write the DNG"));
        let tiff_enc = ExportError::encode("TIFF", "bad").labelled("DNG");
        assert_eq!(tiff_enc.to_string(), "Encoding the DNG failed: bad.");
        let other = ExportError::encode("XMP", "bad").labelled("DNG");
        assert_eq!(other.to_string(), "Encoding the XMP failed: bad.");
    }
}
