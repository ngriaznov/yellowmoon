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
    ///
    /// `Display` prints one of three catalogue sentences ([`MESSAGE_EMPTY`], [`MESSAGE_CUT_SHORT`],
    /// [`MESSAGE_DAMAGED`]) chosen from `detail`; `detail` itself is for the log, `--json` and `-v`.
    #[error("{}", corrupt_message(.detail))]
    Corrupt {
        /// What was wrong, for the log and for bug reports. Never shown to the person.
        detail: String,
    },
    /// The file could not be opened or read. `Display` prints a catalogue sentence chosen from
    /// the error kind; the underlying error stays available as the source.
    #[error("{}", io_message(.0))]
    Io(#[source] std::io::Error),
}

/// The `detail` of a [`RawError::Corrupt`] for a file of length 0.
pub const EMPTY: &str = "the file is empty";
/// The `detail` of a [`RawError::Corrupt`] for a file that is shorter than its directories say:
/// a read past the end, raw data, a tile, a value or a directory beyond the end of the file.
pub const CUT_SHORT: &str = "the raw data runs past the end of the file (the file is cut short)";

/// Sentence for an empty file.
pub const MESSAGE_EMPTY: &str = "This file is empty. Copy it from the card again, or skip it.";
/// Sentence for a file that ends before its picture does.
pub const MESSAGE_CUT_SHORT: &str = "This file is incomplete: it ends before its picture does. Copy it from the card again, or skip it.";
/// Sentence for any other damage.
pub const MESSAGE_DAMAGED: &str = "This file is damaged. Copy it from the card again, or skip it.";
/// Sentence for a file that is gone (`NotFound`).
pub const MESSAGE_NOT_FOUND: &str = "Yellowmoon can't find this file any more. It may have been moved, renamed or its drive disconnected.";
/// Sentence for a refused open (`EACCES`, and `PermissionDenied` off macOS).
pub const MESSAGE_DENIED: &str =
    "Yellowmoon isn't allowed to open this file. Check its permissions.";
/// Sentence for a macOS privacy block (`EPERM` on macOS).
pub const MESSAGE_MACOS_BLOCKED: &str = "macOS didn't let Yellowmoon open this file. Allow access in System Settings, under Privacy & Security.";
/// Sentence for every other I/O failure.
pub const MESSAGE_UNREADABLE: &str =
    "Yellowmoon couldn't read this file. Check that the drive is connected, then try again.";

/// The sentence for [`RawError::Corrupt`].
fn corrupt_message(detail: &str) -> &'static str {
    match detail {
        EMPTY => MESSAGE_EMPTY,
        CUT_SHORT => MESSAGE_CUT_SHORT,
        _ => MESSAGE_DAMAGED,
    }
}

/// The sentence for [`RawError::Io`], chosen from the error kind.
fn io_message(error: &std::io::Error) -> &'static str {
    match error.kind() {
        std::io::ErrorKind::NotFound => MESSAGE_NOT_FOUND,
        std::io::ErrorKind::PermissionDenied => {
            // EPERM (1) on macOS is the privacy block (TCC); EACCES (13) is plain permissions.
            if cfg!(target_os = "macos") && error.raw_os_error() == Some(1) {
                MESSAGE_MACOS_BLOCKED
            } else {
                MESSAGE_DENIED
            }
        }
        _ => MESSAGE_UNREADABLE,
    }
}

impl RawError {
    /// What was wrong, for the log, `--json` and `-v`: the `detail` of a damaged file or the
    /// underlying I/O error. `None` for the variants whose sentence says it all.
    pub fn detail(&self) -> Option<String> {
        match self {
            Self::Corrupt { detail } => Some(detail.clone()),
            Self::Io(error) => Some(error.to_string()),
            _ => None,
        }
    }

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
            Self::corrupt(CUT_SHORT)
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
    fn corrupt_prints_a_catalogue_sentence_and_keeps_the_detail() {
        let error = RawError::corrupt("the entropy-coded data ends before the image does");
        assert_eq!(error.to_string(), MESSAGE_DAMAGED);
        assert_eq!(
            error.detail().as_deref(),
            Some("the entropy-coded data ends before the image does")
        );
        assert_eq!(error.kind(), "corrupt");
        assert_eq!(RawError::corrupt(EMPTY).to_string(), MESSAGE_EMPTY);
        assert_eq!(RawError::corrupt(CUT_SHORT).to_string(), MESSAGE_CUT_SHORT);
    }

    #[test]
    fn io_sentence_follows_the_kind() {
        use std::io::{Error, ErrorKind};
        let text = |e: Error| RawError::Io(e).to_string();
        assert_eq!(text(Error::from(ErrorKind::NotFound)), MESSAGE_NOT_FOUND);
        assert_eq!(
            text(Error::from(ErrorKind::PermissionDenied)),
            MESSAGE_DENIED
        );
        assert_eq!(text(Error::from_raw_os_error(13)), MESSAGE_DENIED);
        assert_eq!(text(Error::from(ErrorKind::TimedOut)), MESSAGE_UNREADABLE);
        assert_eq!(text(Error::from_raw_os_error(5)), MESSAGE_UNREADABLE);
        #[cfg(target_os = "macos")]
        assert_eq!(text(Error::from_raw_os_error(1)), MESSAGE_MACOS_BLOCKED);
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
