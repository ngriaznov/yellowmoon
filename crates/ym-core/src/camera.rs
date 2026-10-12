//! Camera match for one photo: `ym-profiles` detection composed with `ym-raw`'s decode kind, the
//! REDUCED flag of Lossless M and S, the STAND-IN flag and the copy of the camera card
//! (`docs/03-camera-profiles.md` section Detection, `docs/04-raw-decoding.md` section Lossless M
//! and S, `docs/11-ui.md` Camera card and Automatic card).
//!
//! `ym-profiles` does not know about the decoder and `ym-raw` does not know about profiles; this is
//! where the two meet, so the app and `ym info` say the same thing.

use serde::{Deserialize, Serialize};
use ym_profiles::detect::{CameraCopy, CameraMatch, MatchKind};
use ym_profiles::registry::ProfileStore;
use ym_raw::error::RawError;
use ym_raw::layout::DecodeKind;
use ym_raw::metadata::RawMetadata;

/// The note under the REDUCED tag of a Lossless M or S file (`docs/04`).
pub const REDUCED_NOTE: &str = "This file was saved in the camera as Lossless M or S. The camera has already processed it, so colour emulation is approximate. Use Lossless L or uncompressed for best results.";

/// The camera of one photo: who the body is, which profile it gets and why, and how its raw data
/// is stored.
#[derive(Debug, Clone, PartialEq)]
pub struct PhotoCamera {
    /// The detection result (`ym_profiles::detect`).
    pub matched: CameraMatch,
    /// Which decoder reads the file (`ym_raw`).
    pub decode: DecodeKind,
    /// Lossless M or S: camera-processed RGB, no demosaic, colour emulation approximate.
    pub reduced: bool,
    /// The profile is a stand-in (not measured on this body): the STAND-IN tag.
    pub stand_in: bool,
    /// The display copy of the Camera and Automatic cards.
    pub copy: CameraCopy,
}

/// The `camera` object of `PhotoSummary` in `src/api/types.ts`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CameraInfo {
    /// Body as shown in the UI: `Sony A1`.
    pub body: String,
    /// The camera's own model string: `ILCE-1`.
    pub model: String,
    /// Sony's model ID, when the file has one.
    pub model_id: Option<u16>,
    /// Display name of the profile used: `A7R IV`.
    pub profile: String,
    /// Profile id: `sony-ilce-7rm4`.
    pub profile_id: String,
    /// Profile version (semver).
    pub profile_version: String,
    /// How the profile was chosen; anything but `exact` gets the STAND-IN tag.
    pub match_kind: MatchKind,
}

impl CameraInfo {
    /// The camera of a file that could not be opened: nothing is known.
    #[must_use]
    pub fn unknown() -> Self {
        Self {
            body: String::new(),
            model: String::new(),
            model_id: None,
            profile: String::new(),
            profile_id: String::new(),
            profile_version: String::new(),
            match_kind: MatchKind::Default,
        }
    }
}

/// The RAW tab's Camera card: Detected, Profile with the STAND-IN tag, the explanation, and the
/// REDUCED tag with its note for Lossless M and S.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CameraCard {
    /// `Detected` row: `Sony A1 (ILCE-1)`.
    pub detected: String,
    /// `Profile` row: `A7R IV`.
    pub profile: String,
    /// Show the STAND-IN tag next to the profile.
    pub stand_in: bool,
    /// The explanation under the card; absent for a measured body.
    pub note: Option<String>,
    /// Show the REDUCED tag.
    pub reduced: bool,
    /// The note under the REDUCED tag.
    pub reduced_note: Option<String>,
    /// How the raw data is stored: `Lossless L`, `compressed (lossy cRAW)`, ...
    pub raw_mode: String,
    /// When no decoder reads this kind of file yet, the plain-words message (ARW 6.0).
    pub unsupported: Option<String>,
}

impl PhotoCamera {
    /// Matches a probed file against the registry. With `forced_profile` (the CLI's `--profile`)
    /// that profile replaces the detected one and the match is [`MatchKind::Forced`].
    ///
    /// Never fails: an unknown id for `forced_profile` is ignored here (the caller checks the id
    /// when it builds its environment) and the detected profile is used.
    #[must_use]
    pub fn new(store: &ProfileStore, meta: &RawMetadata, forced_profile: Option<&str>) -> Self {
        let detected = store.detect(&meta.make, &meta.model, meta.model_id);
        let matched = match forced_profile {
            Some(id) => store
                .detect_forced(&meta.make, &meta.model, meta.model_id, id)
                .unwrap_or(detected),
            None => detected,
        };
        let copy = store.camera_copy(&matched);
        Self {
            decode: meta.decode,
            reduced: matches!(meta.decode, DecodeKind::LosslessMS { .. }),
            stand_in: matched.is_stand_in(),
            matched,
            copy,
        }
    }

    /// The `camera` object of the photo summary.
    #[must_use]
    pub fn info(&self) -> CameraInfo {
        CameraInfo {
            body: self.matched.body.clone(),
            model: self.matched.model.clone(),
            model_id: self.matched.model_id,
            profile: self.copy.profile.clone(),
            profile_id: self.matched.profile.to_string(),
            profile_version: self.matched.profile_version.to_string(),
            match_kind: self.matched.how.clone(),
        }
    }

    /// The Camera card of the RAW tab.
    #[must_use]
    pub fn card(&self) -> CameraCard {
        CameraCard {
            detected: self.copy.detected.clone(),
            profile: self.copy.profile.clone(),
            stand_in: self.stand_in,
            note: self.copy.note.clone(),
            reduced: self.reduced,
            reduced_note: self.reduced.then(|| REDUCED_NOTE.to_owned()),
            raw_mode: self.decode.describe(),
            unsupported: self.unsupported_message(),
        }
    }

    /// Whether a decoder exists for this kind of file (ARW 6.0 and unrecognised layouts: none).
    #[must_use]
    pub fn is_supported(&self) -> bool {
        !matches!(self.decode, DecodeKind::Arw6 | DecodeKind::Unknown { .. })
    }

    /// The message shown for a file that cannot be decoded yet, the same text `ym-raw` returns
    /// when asked to decode it (`docs/04` section Errors).
    #[must_use]
    pub fn unsupported_message(&self) -> Option<String> {
        (!self.is_supported()).then(|| {
            RawError::Unsupported {
                camera: if self.matched.model.is_empty() {
                    self.matched.make.clone()
                } else {
                    self.matched.model.clone()
                },
                variant: self.decode,
            }
            .to_string()
        })
    }
}
