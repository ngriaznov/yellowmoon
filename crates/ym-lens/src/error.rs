//! The errors of the lens crate.

use std::path::PathBuf;

/// Everything that can go wrong when loading lens data, building a plan or running a pass.
///
/// Damaged data are always an error and never a panic.
#[derive(Debug, thiserror::Error)]
pub enum LensError {
    /// A file could not be read.
    #[error("cannot read {path}: {source}")]
    Io {
        /// The file.
        path: PathBuf,
        /// The underlying error.
        source: std::io::Error,
    },
    /// A JSON file is not of the expected shape.
    #[error("{file}: not valid lens JSON: {source}")]
    Json {
        /// The file, relative to the lenses folder.
        file: String,
        /// The parse error.
        source: serde_json::Error,
    },
    /// A `tables.safetensors` file cannot be read.
    #[error("{file}: not a valid safetensors file: {message}")]
    Safetensors {
        /// The file, relative to the lenses folder.
        file: String,
        /// What the reader said.
        message: String,
    },
    /// A file's SHA-256 differs from its line in `SHA256SUMS`.
    #[error("{file}: SHA-256 differs from SHA256SUMS (expected {expected}, found {actual})")]
    Checksum {
        /// The file, relative to the lenses folder.
        file: String,
        /// The digest in `SHA256SUMS`.
        expected: String,
        /// The digest of the file on disk.
        actual: String,
    },
    /// A file the loader reads has no line in `SHA256SUMS`.
    #[error("{file}: no line in SHA256SUMS")]
    MissingChecksum {
        /// The file, relative to the lenses folder.
        file: String,
    },
    /// A file holds a value the format does not allow.
    #[error("{file}: {reason}")]
    Invalid {
        /// The file, relative to the lenses folder.
        file: String,
        /// What is wrong and what the format wants.
        reason: String,
    },
    /// A lens id is not one of the seven lenses.
    #[error("unknown lens {0:?}")]
    UnknownLens(String),
    /// A lens the caller asked for is not in the library.
    #[error("the library has no lens {0:?}")]
    MissingLens(String),
    /// The camera's spectral curves cannot be projected.
    #[error("camera response: {0}")]
    Camera(String),
    /// An amount is not a number in 0 to 100.
    #[error("lens amount {name} must be a number from 0 to 100, found {value}")]
    Amount {
        /// The field.
        name: &'static str,
        /// The offending value.
        value: f64,
    },
    /// A frame, plane or rectangle does not fit the plan.
    #[error("frame mismatch: {0}")]
    Frame(String),
    /// The input image does not cover the pixels a pass needs (the output grown by `halo_px`).
    #[error("the input image does not cover the {needed} the pass needs; it holds {have}")]
    InputTooSmall {
        /// The rectangle the pass needs, as text.
        needed: String,
        /// The rectangle the input holds, as text.
        have: String,
    },
    /// The caller's cancel callback returned true.
    #[error("cancelled")]
    Cancelled,
}

impl LensError {
    /// A [`LensError::Invalid`] for `file`.
    pub(crate) fn invalid(file: &str, reason: impl Into<String>) -> Self {
        LensError::Invalid {
            file: file.to_owned(),
            reason: reason.into(),
        }
    }
}
