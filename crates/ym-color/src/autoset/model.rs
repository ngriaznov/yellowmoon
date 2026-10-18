//! The automatic settings model (`docs/18-auto-settings.md` §4.2): an ensemble of small perceptrons, each giving a key
//! and a contrast with a log-variance per head. The file is read by [`super::format`]; this module runs it.
//!
//! A member standardises the features, runs its layers (`tanh` on the hidden layers, the last one linear) and returns
//! `[key mean, key log-variance, contrast mean, contrast log-variance]`. The ensemble takes the mean of the members'
//! means, and the variance of a head is the mean of the members' variances plus the population variance of their
//! means (the mixture variance). Every sum runs in member order.

use std::path::Path;

use super::AutosetError;
use super::features::{FEATURE_COUNT, FEATURE_NAMES};
use super::format::{self, Layer};

/// The clamp of a member's log-variance before the exponential (`docs/18` §4.4).
const LOGVAR_LIMIT: f64 = 20.0;

/// Outputs of a member: key mean, key log-variance, contrast mean, contrast log-variance.
const MEMBER_OUTPUTS: usize = 4;

/// The prior standard deviations of the two heads (`prior_sigma` of the file, `docs/18` §4.4): the spread a
/// prediction is shrunk against.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PriorSigma {
    /// `sigma0` of the key target, in stops (0.75 in v1).
    pub key_target: f64,
    /// `sigma0` of the contrast, in contrast units (10 in v1).
    pub contrast: f64,
}

/// The ensemble's prediction for one photo: each head's mean and the variance of the mixture.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Prediction {
    /// Where the key should sit after exposure, in stops over middle grey (`key_target`).
    pub key_target: f64,
    /// The variance of `key_target`: the mean of the members' variances plus the variance of their means.
    pub key_var: f64,
    /// The contrast the model asks for, in the contrast units of the C9 stage (before the bound of [`apply`](super::apply)).
    pub contrast: f64,
    /// The variance of `contrast`, by the same mixture rule.
    pub contrast_var: f64,
}

/// A loaded automatic settings model: the standardisation, the prior, the bound of the contrast and the members.
#[derive(Debug, Clone)]
pub struct AutosetModel {
    version: String,
    sha256: String,
    input_mean: [f64; FEATURE_COUNT],
    input_std: [f64; FEATURE_COUNT],
    prior_sigma: PriorSigma,
    contrast_limit: f64,
    members: Vec<Vec<Layer>>,
}

impl AutosetModel {
    /// Reads a model file (JSON text) and validates it, including its SHA-256 (`docs/18` §4.3).
    ///
    /// # Errors
    ///
    /// [`AutosetError::Json`], [`AutosetError::InvalidModel`] or [`AutosetError::HashMismatch`].
    pub fn from_json(text: &str) -> Result<Self, AutosetError> {
        let parsed = format::parse(text)?;
        Ok(Self {
            version: parsed.version,
            sha256: parsed.sha256,
            input_mean: parsed.input_mean,
            input_std: parsed.input_std,
            prior_sigma: parsed.prior_sigma,
            contrast_limit: parsed.contrast_limit,
            members: parsed.members,
        })
    }

    /// Reads a model file from disk (`data/models/autoset/autoset.json` in the repository).
    ///
    /// # Errors
    ///
    /// [`AutosetError::Io`] when the file cannot be read, and those of [`AutosetModel::from_json`].
    pub fn load(path: &Path) -> Result<Self, AutosetError> {
        let text = std::fs::read_to_string(path).map_err(|e| AutosetError::Io {
            path: path.display().to_string(),
            reason: e.to_string(),
        })?;
        Self::from_json(&text)
    }

    /// The model's version string (`model_version`), which exports record as the `autoset` setting.
    #[must_use]
    pub fn version(&self) -> &str {
        &self.version
    }

    /// SHA-256 of the tensors (lowercase hex), as recorded in the file and verified at load.
    #[must_use]
    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    /// The feature names the model reads, in the order of its input: [`FEATURE_NAMES`].
    #[must_use]
    pub fn feature_names(&self) -> &'static [&'static str] {
        &FEATURE_NAMES
    }

    /// The prior standard deviations of the two heads.
    #[must_use]
    pub fn prior_sigma(&self) -> PriorSigma {
        self.prior_sigma
    }

    /// The bound of the contrast head, in contrast units (`contrast_limit`, 30 in v1).
    #[must_use]
    pub fn contrast_limit(&self) -> f64 {
        self.contrast_limit
    }

    /// The number of members of the ensemble (5 in v1).
    #[must_use]
    pub fn member_count(&self) -> usize {
        self.members.len()
    }

    /// Every member's output for the features `x`: `[key mean, key log-variance, contrast mean, contrast log-variance]`,
    /// the raw network outputs (the log-variances are clamped only when they become variances).
    #[must_use]
    pub fn member_outputs(&self, x: &[f64; FEATURE_COUNT]) -> Vec<[f64; MEMBER_OUTPUTS]> {
        let z: [f64; FEATURE_COUNT] =
            std::array::from_fn(|i| (x[i] - self.input_mean[i]) / self.input_std[i]);
        self.members
            .iter()
            .map(|layers| run_member(layers, &z))
            .collect()
    }

    /// The ensemble's prediction for the features `x` (`docs/18` §4.2).
    #[must_use]
    pub fn predict(&self, x: &[f64; FEATURE_COUNT]) -> Prediction {
        let outputs = self.member_outputs(x);
        let key_heads: Vec<(f64, f64)> = outputs.iter().map(|o| (o[0], o[1])).collect();
        let contrast_heads: Vec<(f64, f64)> = outputs.iter().map(|o| (o[2], o[3])).collect();
        let (key_target, key_var) = combine(&key_heads);
        let (contrast, contrast_var) = combine(&contrast_heads);
        Prediction {
            key_target,
            key_var,
            contrast,
            contrast_var,
        }
    }
}

/// One member: `tanh` on every layer but the last, which is linear.
fn run_member(layers: &[Layer], input: &[f64]) -> [f64; MEMBER_OUTPUTS] {
    let mut activation = input.to_vec();
    for (k, layer) in layers.iter().enumerate() {
        let linear = k + 1 == layers.len();
        activation = layer
            .w
            .chunks_exact(layer.cols)
            .zip(&layer.b)
            .map(|(row, bias)| {
                let z = row.iter().zip(&activation).map(|(w, x)| w * x).sum::<f64>() + bias;
                if linear { z } else { z.tanh() }
            })
            .collect();
    }
    let mut out = [0.0; MEMBER_OUTPUTS];
    for (o, v) in out.iter_mut().zip(&activation) {
        *o = *v;
    }
    out
}

/// The ensemble of one head from the members' `(mean, log-variance)` pairs: the mean of the means, and the mean of the
/// variances plus the population variance of the means. The variances are `exp` of the log-variance clamped to
/// [`LOGVAR_LIMIT`].
fn combine(members: &[(f64, f64)]) -> (f64, f64) {
    let n = members.len() as f64;
    let mean = members.iter().map(|(m, _)| m).sum::<f64>() / n;
    let mean_variance = members
        .iter()
        .map(|(_, logvar)| logvar.clamp(-LOGVAR_LIMIT, LOGVAR_LIMIT).exp())
        .sum::<f64>()
        / n;
    let spread = members
        .iter()
        .map(|(m, _)| (m - mean) * (m - mean))
        .sum::<f64>()
        / n;
    (mean, mean_variance + spread)
}
