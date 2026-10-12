//! Typed decode errors (`Unsupported`, `NotArw`, `Corrupt`) with plain-language messages (`docs/04-raw-decoding.md` section Errors).
//!
//! Every failure of the probe or a decoder is a [`RawError`]. The `Display` text of each variant is
//! the message the app shows next to the file, so it says what went wrong and what to do next. A
//! failing file never blocks the session; callers keep going with the next one.

use crate::layout::DecodeKind;

/// Why a file could not be probed or decoded.
#[derive(Debug, thiserror::Error)]
pub enum RawError {
    /// A known variant that Yellowmoon does not decode yet (ARW 6.0 until the Sony gate clears).
    #[error("{}", unsupported_message(.camera, .variant))]
    Unsupported {
        /// The camera that wrote the file (the `Model` string, or the make when there is none).
        camera: String,
        /// The recognised but unsupported way the raw data is stored.
        variant: DecodeKind,
    },
    /// SRF, SR2 or a file that is not an ARW at all (not a TIFF, not a Sony raw).
    #[error("This isn't an ARW file. Yellowmoon opens Sony ARW files only.")]
    NotArw,
    /// A pixel-shift ARQ file (four samples per pixel).
    #[error("Pixel-shift ARQ files aren't supported. Open the individual ARW frames instead.")]
    Arq,
    /// The file is truncated or malformed.
    #[error(
        "This file is damaged or incomplete ({detail}). Copy it from the card again, or skip it."
    )]
    Corrupt {
        /// What was wrong, for the log and for bug reports.
        detail: String,
    },
    /// The file could not be opened or read.
    #[error(
        "Yellowmoon couldn't read this file ({0}). Check that it still exists and that you can open it."
    )]
    Io(#[source] std::io::Error),
}

impl RawError {
    /// Builds a [`RawError::Corrupt`] from any message.
    pub fn corrupt(detail: impl Into<String>) -> Self {
        Self::Corrupt {
            detail: detail.into(),
        }
    }

    /// A short machine-readable name of the error class, as used in `--json` output
    /// (`unsupported`, `not_arw`, `arq`, `corrupt` or `io`).
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Unsupported { .. } => "unsupported",
            Self::NotArw => "not_arw",
            Self::Arq => "arq",
            Self::Corrupt { .. } => "corrupt",
            Self::Io(_) => "io",
        }
    }
}

impl From<std::io::Error> for RawError {
    /// A read past the end of the file means the file is truncated, so it is reported as
    /// [`RawError::Corrupt`]; every other I/O failure stays [`RawError::Io`].
    fn from(error: std::io::Error) -> Self {
        if error.kind() == std::io::ErrorKind::UnexpectedEof {
            Self::corrupt("the file ends before its data does")
        } else {
            Self::Io(error)
        }
    }
}

/// The message for [`RawError::Unsupported`].
fn unsupported_message(camera: &str, variant: &DecodeKind) -> String {
    match variant {
        DecodeKind::Arw6 => {
            "ARW 6.0 compressed files from the A7 V and A7R VI aren't supported yet. \
             Lossless files from these cameras work. We're working on it."
                .to_owned()
        }
        other => format!(
            "Yellowmoon can't open {} files from the {camera} yet. \
             Save the photo in another raw mode in the camera, or wait for an update.",
            other.describe()
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::ChromaSubsampling;

    #[test]
    fn arw6_message_is_the_one_in_the_spec() {
        let error = RawError::Unsupported {
            camera: "ILCE-7M5".to_owned(),
            variant: DecodeKind::Arw6,
        };
        assert_eq!(
            error.to_string(),
            "ARW 6.0 compressed files from the A7 V and A7R VI aren't supported yet. \
             Lossless files from these cameras work. We're working on it."
        );
        assert_eq!(error.kind(), "unsupported");
    }

    #[test]
    fn not_arw_and_arq_messages_are_the_ones_in_the_spec() {
        assert_eq!(
            RawError::NotArw.to_string(),
            "This isn't an ARW file. Yellowmoon opens Sony ARW files only."
        );
        assert_eq!(
            RawError::Arq.to_string(),
            "Pixel-shift ARQ files aren't supported. Open the individual ARW frames instead."
        );
        assert_eq!(RawError::NotArw.kind(), "not_arw");
        assert_eq!(RawError::Arq.kind(), "arq");
    }

    #[test]
    fn other_unsupported_variants_name_the_camera_and_the_mode() {
        let error = RawError::Unsupported {
            camera: "ILCE-7RM5".to_owned(),
            variant: DecodeKind::LosslessMS {
                subsampling: ChromaSubsampling::Yuv420,
            },
        };
        let text = error.to_string();
        assert!(text.contains("ILCE-7RM5"), "{text}");
        assert!(text.contains("Lossless M or S"), "{text}");
    }

    #[test]
    fn corrupt_message_carries_the_detail_and_a_next_step() {
        let error = RawError::corrupt("raw data ends beyond the end of the file");
        let text = error.to_string();
        assert!(
            text.contains("raw data ends beyond the end of the file"),
            "{text}"
        );
        assert!(text.contains("Copy it from the card again"), "{text}");
        assert_eq!(error.kind(), "corrupt");
    }

    #[test]
    fn unexpected_eof_becomes_corrupt_and_other_io_errors_stay_io() {
        let eof = std::io::Error::from(std::io::ErrorKind::UnexpectedEof);
        assert!(matches!(RawError::from(eof), RawError::Corrupt { .. }));
        let denied = std::io::Error::from(std::io::ErrorKind::PermissionDenied);
        let error = RawError::from(denied);
        assert!(matches!(error, RawError::Io(_)));
        assert_eq!(error.kind(), "io");
    }
}
