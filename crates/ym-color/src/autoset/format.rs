//! The model file (`data/models/autoset/autoset.json`, `docs/18-auto-settings.md` §4.3): reading, validation and the
//! SHA-256 check. It mirrors the illuminant model's file (`crate::illum::format`).
//!
//! The file is JSON. The standardisation and the prior are plain numbers; each tensor is `{"shape": [...],
//! "data_b64": "..."}`, the data being base64 of little-endian `f32` in row-major order. The SHA-256 covers the bytes
//! of every tensor in order: members in order, then layers in order, `w` before `b`. The checks run in this order:
//! the header, the shapes and the values, then the digest; so a file with a bad shape reports the shape.

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use serde::Deserialize;
use sha2::{Digest, Sha256};

use super::AutosetError;
use super::features::{FEATURE_COUNT, FEATURE_NAMES};
use super::model::PriorSigma;

/// Value of the `format` field.
const FORMAT: &str = "yellowmoon-autoset";
/// Value of the `format_version` field this reader understands.
const FORMAT_VERSION: u64 = 1;
/// Value of `activation`: the non-linearity of the hidden layers.
const ACTIVATION: &str = "tanh";
/// Value of `outputs`: the names of a member's four outputs, in order.
const OUTPUTS: [&str; 4] = ["key_target", "key_logvar", "contrast", "contrast_logvar"];
/// Outputs of a member's last layer.
const N_OUTPUTS: usize = OUTPUTS.len();
/// Layers of a member: at least one hidden layer and the linear output layer.
const MIN_LAYERS: usize = 2;

#[derive(Debug, Deserialize)]
struct ModelFile {
    format: String,
    format_version: u64,
    model_version: String,
    feature_names: Vec<String>,
    input_mean: Vec<f64>,
    input_std: Vec<f64>,
    activation: String,
    outputs: Vec<String>,
    prior_sigma: PriorSigmaFile,
    contrast_limit: f64,
    members: Vec<MemberFile>,
    sha256: String,
}

#[derive(Debug, Deserialize)]
struct PriorSigmaFile {
    key_target: f64,
    contrast: f64,
}

#[derive(Debug, Deserialize)]
struct MemberFile {
    layers: Vec<LayerFile>,
}

#[derive(Debug, Deserialize)]
struct LayerFile {
    w: TensorFile,
    b: TensorFile,
}

#[derive(Debug, Deserialize)]
struct TensorFile {
    shape: Vec<usize>,
    data_b64: String,
    /// Optional; when present it must be `f32le`.
    dtype: Option<String>,
}

/// One layer of a member: `w` is `rows x cols` row-major and `b` has `rows` values, as f64 converted from f32.
#[derive(Debug, Clone)]
pub(super) struct Layer {
    /// Inputs of the layer (`cols` of `w`).
    pub(super) cols: usize,
    /// `rows x cols` weights, row-major.
    pub(super) w: Vec<f64>,
    /// One bias per output row.
    pub(super) b: Vec<f64>,
}

/// A validated model file: what [`super::AutosetModel`] keeps.
#[derive(Debug, Clone)]
pub(super) struct Parsed {
    pub(super) version: String,
    pub(super) sha256: String,
    pub(super) input_mean: [f64; FEATURE_COUNT],
    pub(super) input_std: [f64; FEATURE_COUNT],
    pub(super) prior_sigma: PriorSigma,
    pub(super) contrast_limit: f64,
    pub(super) members: Vec<Vec<Layer>>,
}

fn bad(reason: impl Into<String>) -> AutosetError {
    AutosetError::InvalidModel(reason.into())
}

/// `values` as a fixed array of finite numbers.
fn fixed(values: &[f64], name: &str) -> Result<[f64; FEATURE_COUNT], AutosetError> {
    let array: [f64; FEATURE_COUNT] = values.try_into().map_err(|_| {
        bad(format!(
            "{name} needs {FEATURE_COUNT} values, found {}",
            values.len()
        ))
    })?;
    if !array.iter().all(|v| v.is_finite()) {
        return Err(bad(format!("{name} has a value that is not finite")));
    }
    Ok(array)
}

/// The two dimensions of a weight tensor, each at least 1.
fn dims2(shape: &[usize], name: &str) -> Result<(usize, usize), AutosetError> {
    match *shape {
        [rows, cols] if rows >= 1 && cols >= 1 => Ok((rows, cols)),
        _ => Err(bad(format!(
            "{name} has shape {shape:?}, expected two dimensions of at least 1"
        ))),
    }
}

/// Decodes one tensor of the stated shape: base64 of `f32le`, the byte count of the shape, every value finite. Returns
/// the values as f64 and the bytes, which the digest covers.
fn tensor(
    t: &TensorFile,
    shape: &[usize],
    name: &str,
) -> Result<(Vec<f64>, Vec<u8>), AutosetError> {
    if t.dtype.as_deref().is_some_and(|dtype| dtype != "f32le") {
        return Err(bad(format!(
            "tensor {name} has dtype {:?}, expected f32le",
            t.dtype.as_deref().unwrap_or_default()
        )));
    }
    let count = shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| bad(format!("tensor {name} is too large")))?;
    let bytes = STANDARD
        .decode(t.data_b64.as_bytes())
        .map_err(|e| bad(format!("tensor {name} is not valid base64: {e}")))?;
    let expected = count
        .checked_mul(4)
        .ok_or_else(|| bad(format!("tensor {name} is too large")))?;
    if bytes.len() != expected {
        return Err(bad(format!(
            "tensor {name} holds {} bytes, expected {expected} for shape {shape:?}",
            bytes.len()
        )));
    }
    let values: Vec<f64> = bytes
        .chunks_exact(4)
        .map(|b| f64::from(f32::from_le_bytes([b[0], b[1], b[2], b[3]])))
        .collect();
    if !values.iter().all(|v| v.is_finite()) {
        return Err(bad(format!("tensor {name} has a value that is not finite")));
    }
    Ok((values, bytes))
}

/// Parses and validates a model file (`docs/18` §4.3).
pub(super) fn parse(text: &str) -> Result<Parsed, AutosetError> {
    let file: ModelFile =
        serde_json::from_str(text).map_err(|e| AutosetError::Json(e.to_string()))?;
    if file.format != FORMAT || file.format_version != FORMAT_VERSION {
        return Err(bad(format!(
            "format {:?} version {} is not {FORMAT} version {FORMAT_VERSION}",
            file.format, file.format_version
        )));
    }
    if file.model_version.is_empty() {
        return Err(bad("model_version is empty"));
    }
    if file.feature_names != FEATURE_NAMES {
        return Err(bad(format!(
            "feature_names must be the {FEATURE_COUNT} names of the feature vector, in order"
        )));
    }
    if file.activation != ACTIVATION {
        return Err(bad(format!(
            "activation {:?} is not {ACTIVATION}",
            file.activation
        )));
    }
    if file.outputs != OUTPUTS {
        return Err(bad(format!("outputs must be {OUTPUTS:?}")));
    }
    let input_mean = fixed(&file.input_mean, "input_mean")?;
    let input_std = fixed(&file.input_std, "input_std")?;
    if let Some(i) = input_std.iter().position(|s| *s <= 0.0) {
        return Err(bad(format!(
            "input_std must be positive, found {} for {}",
            input_std[i], FEATURE_NAMES[i]
        )));
    }
    let positive = |name: &str, v: f64| {
        if v.is_finite() && v > 0.0 {
            Ok(v)
        } else {
            Err(bad(format!(
                "{name} must be a positive finite number, found {v}"
            )))
        }
    };
    let prior_sigma = PriorSigma {
        key_target: positive("prior_sigma.key_target", file.prior_sigma.key_target)?,
        contrast: positive("prior_sigma.contrast", file.prior_sigma.contrast)?,
    };
    let contrast_limit = positive("contrast_limit", file.contrast_limit)?;
    if file.members.is_empty() {
        return Err(bad("the model needs at least one member"));
    }

    // The digest covers the tensors in the order they are read: members, layers, then `w` before `b`.
    let mut hasher = Sha256::new();
    let mut members = Vec::with_capacity(file.members.len());
    for (mi, member) in file.members.iter().enumerate() {
        if member.layers.len() < MIN_LAYERS {
            return Err(bad(format!(
                "member {mi} has {} layers, at least {MIN_LAYERS} are needed (hidden and output)",
                member.layers.len()
            )));
        }
        let mut layers = Vec::with_capacity(member.layers.len());
        let mut inputs = FEATURE_COUNT;
        for (li, layer) in member.layers.iter().enumerate() {
            let name = format!("member {mi} layer {li}");
            let (rows, cols) = dims2(&layer.w.shape, &format!("{name} w"))?;
            if cols != inputs {
                return Err(bad(format!(
                    "{name} takes {cols} inputs but the layer before it gives {inputs}"
                )));
            }
            if layer.b.shape.as_slice() != [rows] {
                return Err(bad(format!(
                    "{name} b has shape {:?}, expected [{rows}]",
                    layer.b.shape
                )));
            }
            let (w, w_bytes) = tensor(&layer.w, &[rows, cols], &format!("{name} w"))?;
            let (b, b_bytes) = tensor(&layer.b, &[rows], &format!("{name} b"))?;
            hasher.update(&w_bytes);
            hasher.update(&b_bytes);
            layers.push(Layer { cols, w, b });
            inputs = rows;
        }
        if inputs != N_OUTPUTS {
            return Err(bad(format!(
                "member {mi} ends with {inputs} outputs, expected {N_OUTPUTS}"
            )));
        }
        members.push(layers);
    }

    let digest: String = hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    if !digest.eq_ignore_ascii_case(&file.sha256) {
        return Err(AutosetError::HashMismatch {
            expected: file.sha256,
            found: digest,
        });
    }
    Ok(Parsed {
        version: file.model_version,
        sha256: digest,
        input_mean,
        input_std,
        prior_sigma,
        contrast_limit,
        members,
    })
}
