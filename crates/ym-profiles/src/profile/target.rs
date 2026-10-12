//! The emulation target description, `data/alexa/target.json` (`docs/03-camera-profiles.md`, the
//! paragraph on the ALEXA target; `schema/target.schema.json`).
//!
//! The target is the ALEXA ALEV III spectral sensitivity **estimate** the sensor maps are fitted
//! against. The file names the curves, records where they came from and how well they validated, and
//! carries the caveats the app shows wherever the target is described ("honest about estimates",
//! `docs/00-overview.md`).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::ProfileError;

/// The variant of the estimate that was shipped.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecommendedVariant {
    /// Prior used (`all`).
    pub prior: String,
    /// Number of principal components.
    #[serde(rename = "K")]
    pub k: u32,
    /// Regularisation weight.
    pub lam: f64,
    /// Matrix convention (`rta`).
    pub conv: String,
}

/// Published validation numbers of the estimate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublishedValidation {
    /// Median relative RMS error of the curves, per cent of peak.
    pub rfse_median_pct: f64,
    /// 90th percentile of the same.
    pub rfse_p90_pct: f64,
    /// Median emulation error, ΔE00.
    #[serde(rename = "emulation_dE00_median")]
    pub emulation_de00_median: f64,
    /// Median over cameras of the 95th percentile emulation error, ΔE00.
    #[serde(rename = "emulation_dE00_p95_median_over_cameras")]
    pub emulation_de00_p95_median_over_cameras: f64,
}

/// How the estimate was validated.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetValidation {
    /// The method (leave-one-camera-out over the measured cameras).
    pub method: String,
    /// The published numbers.
    pub published: PublishedValidation,
    /// Allowed deviation when the validation is reproduced in CI.
    pub tolerance: f64,
    /// The gate in `docs/13`.
    pub gate: String,
}

/// Provenance of the target.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetProvenance {
    /// One-sentence description.
    pub summary: String,
    /// How the curves were estimated.
    pub method: String,
    /// Inputs and their licences.
    pub inputs: Vec<String>,
    /// Licence of the result.
    pub licence: String,
    /// Script that produced the curves.
    pub fit_script: String,
    /// The shipped variant.
    pub recommended_variant: RecommendedVariant,
    /// Validation of the estimate.
    pub validation: TargetValidation,
    /// What the estimate cannot tell.
    pub caveats: Vec<String>,
    /// SHA-256 of each file of the target, by file name.
    pub sha256: BTreeMap<String, String>,
}

/// `data/alexa/target.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Target {
    /// Target id (`alexa-alev3-estimate`).
    pub id: String,
    /// Semver of the target.
    pub version: semver::Version,
    /// Name for display.
    pub display_name: String,
    /// `estimate` or `measurement`.
    pub kind: TargetKind,
    /// Curves in rawtoaces-data format, relative to the data root.
    pub spectral_curves: String,
    /// Ensemble statistics, relative to the data root.
    pub ensemble_csv: String,
    /// The 24-variant ensemble, relative to the data root.
    pub variants_csv: String,
    /// Where the numbers came from.
    pub provenance: TargetProvenance,
}

/// Whether the target curves were measured or inferred.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TargetKind {
    /// Inferred from published matrices.
    Estimate,
    /// Measured.
    Measurement,
}

impl Target {
    /// Parses the text of a `target.json`.
    ///
    /// # Errors
    ///
    /// [`ProfileError::Json`] for malformed or mismatching JSON.
    pub fn parse(text: &str) -> Result<Self, ProfileError> {
        Self::parse_labelled(text, Path::new("<text>"))
    }

    fn parse_labelled(text: &str, path: &Path) -> Result<Self, ProfileError> {
        serde_json::from_str(text).map_err(|source| ProfileError::Json {
            path: path.to_path_buf(),
            source,
        })
    }

    /// Reads and parses a `target.json`.
    ///
    /// # Errors
    ///
    /// [`ProfileError::Io`] or [`ProfileError::Json`].
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, ProfileError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|source| ProfileError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        Self::parse_labelled(&text, path)
    }

    /// Whether the curves are an estimate; the UI and the documentation must say so in plain words.
    #[must_use]
    pub fn is_estimate(&self) -> bool {
        self.kind == TargetKind::Estimate
    }

    /// Path of the curve file under the data root (`<data_root>/<spectral_curves>`).
    #[must_use]
    pub fn curves_path(&self, data_root: &Path) -> PathBuf {
        data_root.join(&self.spectral_curves)
    }
}
