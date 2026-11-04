//! Errors of the film engine. Each says what went wrong and, where it can, what to do.

use std::path::PathBuf;

/// Everything that can go wrong building or running a film chain.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum FilmError {
    /// A file could not be read.
    #[error(
        "Cannot read {path}: {source}. Check that the file exists and reinstall the app if it does not."
    )]
    Io {
        /// The file.
        path: PathBuf,
        /// The operating system's reason.
        #[source]
        source: std::io::Error,
    },
    /// A data file is damaged: it does not parse, or a shape or a range is wrong.
    #[error("{file} is damaged: {what}. Reinstall the app to restore it.")]
    Data {
        /// The file name (and folder).
        file: String,
        /// What is wrong with it.
        what: String,
    },
    /// The SHA-256 of a file is not the one in `SHA256SUMS`.
    #[error("{file} does not match its checksum in SHA256SUMS. Reinstall the app to restore it.")]
    Checksum {
        /// The file.
        file: String,
    },
    /// A stock id that is not in the data.
    #[error("There is no film or paper called {0}.")]
    UnknownStock(String),
    /// A stock of the wrong kind for its place (a paper used as a film, and so on).
    #[error("{id} cannot be used as {wanted}.")]
    WrongKind {
        /// The stock.
        id: String,
        /// What it was asked to be.
        wanted: &'static str,
    },
    /// The coupler matrix makes the film's curves non-invertible.
    #[error(
        "The couplers cannot be undone for {stock}: the effective exposure does not rise with the exposure in channel {channel}. Lower the coupler matrix in couplers.json."
    )]
    CouplersNotInvertible {
        /// The film.
        stock: String,
        /// The channel (0 red, 1 green, 2 blue).
        channel: usize,
    },
    /// The neutral solve or the brightness match did not settle.
    #[error("Could not balance {film} on {paper}: {what}.")]
    Calibration {
        /// The film.
        film: String,
        /// The paper.
        paper: String,
        /// What failed.
        what: String,
    },
    /// The paper is viewed under a light the data does not have.
    #[error("{id} is viewed under {light}, which this version does not have.")]
    ViewingLight {
        /// The paper.
        id: String,
        /// The light.
        light: String,
    },
}

impl FilmError {
    pub(crate) fn data(file: impl Into<String>, what: impl Into<String>) -> Self {
        Self::Data {
            file: file.into(),
            what: what.into(),
        }
    }
}
