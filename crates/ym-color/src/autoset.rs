//! Automatic settings, `autoset` v1 (`docs/18-auto-settings.md`): the scene features the model reads, the learned
//! exposure and contrast model, the shrinkage towards the camera-gain rule, and the preferred white.
//!
//! Everything here is a pure function over plain buffers, in f64, on one thread and in a fixed order, so the same
//! inputs give the same bits on every machine (AGENTS rule 8). The parts follow the spec:
//!
//! * [`features`] (§3): the 39 values the model reads, from the metering preview and the scene light;
//! * [`AutosetModel`] (§4.2 and §4.3): the ensemble of small perceptrons and its file;
//! * [`apply`] and [`rule_only`] (§4.4): the prediction shrunk towards the rule by its own variance;
//! * [`preferred_white`] (§5): the user white for an Auto white balance.
//!
//! Reading the model file is the only file access; where the file lives and what to do when it is missing is the
//! caller's choice. Without a usable model the caller uses [`rule_only`] (the camera's gain, no contrast).

mod features;
mod format;
mod model;
mod shrink;
mod white;

pub use features::{FEATURE_COUNT, FEATURE_NAMES, FeatureInputs, Features, features};
pub use model::{AutosetModel, Prediction, PriorSigma};
pub use shrink::{Applied, apply, rule_only};
pub use white::{PreferredWhite, preferred_white};

use crate::exposure::ExposureError;

/// Errors from the automatic settings: a metering preview or scene light that cannot be used, and a model file that
/// cannot be read or is not the model this code expects.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum AutosetError {
    /// The metering preview's buffers do not match its size.
    #[error("the metering preview is not usable: {0}")]
    Preview(#[from] ExposureError),
    /// No pixel of the metering preview has finite channels, so there is nothing to meter.
    #[error("the metering preview has no pixel with finite values")]
    NoUsablePixel,
    /// The mean luminance is not a positive, finite number.
    #[error("the mean luminance {0} must be a positive, finite number")]
    MeanLuminance(f64),
    /// The clip level's luminance is not a positive, finite number.
    #[error("the clip level {0} must be a positive, finite luminance")]
    ClipLevel(f64),
    /// A feature came out not finite, for example from a scene light whose CCT is not positive.
    #[error("the feature {0} is not a finite number")]
    NonFinite(&'static str),
    /// The model file could not be read.
    #[error("cannot read the automatic settings model {path}: {reason}")]
    Io {
        /// The file.
        path: String,
        /// The operating system's message.
        reason: String,
    },
    /// The model file is not valid JSON of the expected layout.
    #[error("the automatic settings model file is not valid: {0}")]
    Json(String),
    /// The model file parsed but its contents are not usable.
    #[error("the automatic settings model is not usable: {0}")]
    InvalidModel(String),
    /// The tensors do not match the SHA-256 recorded in the file.
    #[error(
        "the automatic settings model does not match its SHA-256 (file says {expected}, tensors give {found})"
    )]
    HashMismatch {
        /// The digest in the file.
        expected: String,
        /// The digest of the tensors.
        found: String,
    },
}
