//! The illuminant model file (`data/models/illum/<profile-id>.json`): reading, validation and the SHA-256 check.
//!
//! The file is JSON with readable scalars and the tensors as base64 of little-endian `f32` (the exact bytes the
//! SHA-256 in the file covers, concatenated in `tensor_order`). `training/illum/model.py` writes it.

use std::collections::HashMap;

use serde::Deserialize;
use sha2::{Digest, Sha256};

use super::IllumError;
use super::histogram::{HistogramChannel, HistogramSpec};

/// Value of the `format` field.
pub(crate) const FORMAT: &str = "yellowmoon-illum";
/// Value of the `format_version` field this reader understands.
pub(crate) const FORMAT_VERSION: u64 = 1;
/// Largest histogram size the loader accepts (bins per axis).
pub(crate) const MAX_BINS: usize = 256;
/// Number of illuminant classes (the order of the head's outputs).
pub(crate) const N_CLASSES: usize = 6;
/// The tensors of a model file, in hash order.
pub(crate) const TENSOR_ORDER: [&str; 11] = [
    "filter",
    "bias",
    "w1",
    "b1",
    "w2",
    "b2",
    "feature_mean",
    "feature_scale",
    "conf_w",
    "conf_mean",
    "conf_scale",
];
/// Extra inputs of the confidence regression on top of the class-head features.
pub(crate) const N_CONF_EXTRA: usize = 3;

#[derive(Debug, Deserialize)]
pub(crate) struct ModelFile {
    pub format: String,
    pub format_version: u64,
    pub model_version: String,
    pub profile_id: String,
    pub histogram: HistogramJson,
    pub fit: FitJson,
    pub confidence: ConfidenceJson,
    pub head: HeadJson,
    pub tensors: HashMap<String, TensorJson>,
    pub tensor_order: Vec<String>,
    pub param_count: u64,
    pub sha256: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct HistogramJson {
    pub n_bins: usize,
    pub bin_size: f64,
    pub lo_u: f64,
    pub lo_v: f64,
    pub min_intensity: f64,
    pub min_edge: f64,
    pub channels: Vec<String>,
    pub weight: String,
    pub y_weights: Vec<f64>,
    #[serde(default = "default_max_samples")]
    pub max_samples: usize,
}

fn default_max_samples() -> usize {
    16384
}

#[derive(Debug, Deserialize)]
pub(crate) struct FitJson {
    pub pad_variance: f64,
    pub prior_sigma: f64,
    pub default_uv: [f64; 2],
}

#[derive(Debug, Deserialize)]
pub(crate) struct ConfidenceJson {
    pub kind: String,
    pub n_features: usize,
}

#[derive(Debug, Deserialize)]
pub(crate) struct HeadJson {
    pub classes: Vec<String>,
    pub n_features: usize,
    pub hidden: usize,
    pub temperature: f64,
    pub bv_default: f64,
    pub class_prior: Vec<f64>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct TensorJson {
    pub shape: Vec<usize>,
    pub dtype: String,
    pub data_b64: String,
}

/// The class names in the order of the head's outputs.
pub(crate) const CLASS_NAMES: [&str; N_CLASSES] = [
    "daylight",
    "shade_overcast",
    "tungsten_halogen",
    "fluorescent",
    "led",
    "mixed",
];

fn bad(reason: impl Into<String>) -> IllumError {
    IllumError::InvalidModel(reason.into())
}

/// Decodes standard base64 (with `=` padding).
pub(crate) fn base64_decode(text: &str) -> Result<Vec<u8>, IllumError> {
    let mut out = Vec::with_capacity(text.len() / 4 * 3);
    let mut acc = 0u32;
    let mut bits = 0u32;
    let mut pad = 0usize;
    for c in text.bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            b'=' => {
                pad += 1;
                continue;
            }
            b'\n' | b'\r' | b' ' => continue,
            _ => return Err(bad("tensor data is not valid base64")),
        };
        if pad > 0 {
            return Err(bad("tensor data has characters after the base64 padding"));
        }
        acc = (acc << 6) | u32::from(v);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    Ok(out)
}

fn tensor(
    file: &ModelFile,
    name: &str,
    shape: &[usize],
) -> Result<(Vec<f64>, Vec<u8>), IllumError> {
    let t = file
        .tensors
        .get(name)
        .ok_or_else(|| bad(format!("tensor {name} is missing")))?;
    if t.dtype != "f32le" {
        return Err(bad(format!(
            "tensor {name} has dtype {}, expected f32le",
            t.dtype
        )));
    }
    if t.shape != shape {
        return Err(bad(format!(
            "tensor {name} has shape {:?}, expected {shape:?}",
            t.shape
        )));
    }
    let bytes = base64_decode(&t.data_b64)?;
    let expected: usize = shape.iter().product::<usize>() * 4;
    if bytes.len() != expected {
        return Err(bad(format!(
            "tensor {name} holds {} bytes, expected {expected}",
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

/// A validated model file: the specification and the tensors as `f64`.
#[derive(Debug, Clone)]
pub(crate) struct Parsed {
    pub profile_id: String,
    pub version: String,
    pub spec: HistogramSpec,
    pub pad: f64,
    pub prior_sigma: f64,
    pub default_uv: [f64; 2],
    pub temperature: f64,
    pub bv_default: f64,
    pub class_prior: [f64; N_CLASSES],
    pub hidden: usize,
    pub n_features: usize,
    pub filter: Vec<f64>,
    pub bias: Vec<f64>,
    pub w1: Vec<f64>,
    pub b1: Vec<f64>,
    pub w2: Vec<f64>,
    pub b2: Vec<f64>,
    pub feature_mean: Vec<f64>,
    pub feature_scale: Vec<f64>,
    pub conf_w: Vec<f64>,
    pub conf_mean: Vec<f64>,
    pub conf_scale: Vec<f64>,
    pub sha256: String,
    pub param_count: u64,
}

fn positive(name: &str, v: f64) -> Result<(), IllumError> {
    if v.is_finite() && v > 0.0 {
        Ok(())
    } else {
        Err(bad(format!(
            "{name} must be a positive finite number, found {v}"
        )))
    }
}

/// Parses and validates a model file.
pub(crate) fn parse(text: &str) -> Result<Parsed, IllumError> {
    let file: ModelFile =
        serde_json::from_str(text).map_err(|e| IllumError::Json(e.to_string()))?;
    if file.format != FORMAT || file.format_version != FORMAT_VERSION {
        return Err(bad(format!(
            "format {:?} version {} is not {FORMAT} version {FORMAT_VERSION}",
            file.format, file.format_version
        )));
    }
    if file.tensor_order != TENSOR_ORDER {
        return Err(bad(format!("tensor_order must be {TENSOR_ORDER:?}")));
    }
    let hj = &file.histogram;
    let n = hj.n_bins;
    if !((2..=MAX_BINS).contains(&n) && n.is_power_of_two()) {
        return Err(bad(format!(
            "n_bins {n} must be a power of two from 2 to {MAX_BINS}"
        )));
    }
    positive("bin_size", hj.bin_size)?;
    positive("min_intensity", hj.min_intensity)?;
    positive("min_edge", hj.min_edge)?;
    if !(hj.lo_u.is_finite() && hj.lo_v.is_finite()) {
        return Err(bad("lo_u and lo_v must be finite"));
    }
    if hj.weight != "sqrt_y" || hj.y_weights != [0.25, 0.5, 0.25] {
        return Err(bad(
            "only the sqrt(Y) weight with Y = 0.25 R + 0.5 G + 0.25 B is supported",
        ));
    }
    let channels = hj
        .channels
        .iter()
        .map(|c| match c.as_str() {
            "intensity" => Ok(HistogramChannel::Intensity),
            "edge" => Ok(HistogramChannel::Edge),
            other => Err(bad(format!("unknown histogram channel {other:?}"))),
        })
        .collect::<Result<Vec<_>, _>>()?;
    if channels.is_empty() || channels.len() > 4 {
        return Err(bad("a model needs one to four histogram channels"));
    }
    if hj.max_samples == 0 {
        return Err(bad("max_samples must be at least 1"));
    }
    let c = channels.len();
    let (filter, mut hash_bytes) = tensor(&file, "filter", &[c, n, n])?;
    let (bias, bytes) = tensor(&file, "bias", &[n, n])?;
    hash_bytes.extend(bytes);
    let hd = &file.head;
    if hd.classes != CLASS_NAMES {
        return Err(bad(format!("head classes must be {CLASS_NAMES:?}")));
    }
    if hd.n_features != 11 + 7 * c || hd.hidden == 0 || hd.hidden > 1024 {
        return Err(bad(format!(
            "head has {} features and {} hidden units; {} features are expected for {c} channels",
            hd.n_features,
            hd.hidden,
            11 + 7 * c
        )));
    }
    let (w1, bytes) = tensor(&file, "w1", &[hd.hidden, hd.n_features])?;
    hash_bytes.extend(bytes);
    let (b1, bytes) = tensor(&file, "b1", &[hd.hidden])?;
    hash_bytes.extend(bytes);
    let (w2, bytes) = tensor(&file, "w2", &[N_CLASSES, hd.hidden])?;
    hash_bytes.extend(bytes);
    let (b2, bytes) = tensor(&file, "b2", &[N_CLASSES])?;
    hash_bytes.extend(bytes);
    let (feature_mean, bytes) = tensor(&file, "feature_mean", &[hd.n_features])?;
    hash_bytes.extend(bytes);
    let (feature_scale, bytes) = tensor(&file, "feature_scale", &[hd.n_features])?;
    hash_bytes.extend(bytes);
    if !feature_scale.iter().all(|s| *s > 0.0) {
        return Err(bad("feature_scale must be positive"));
    }
    let n_conf = hd.n_features + N_CONF_EXTRA;
    if file.confidence.kind != "logistic" || file.confidence.n_features != n_conf {
        return Err(bad(format!(
            "confidence must be a logistic regression on {n_conf} features, found {:?} on {}",
            file.confidence.kind, file.confidence.n_features
        )));
    }
    let (conf_w, bytes) = tensor(&file, "conf_w", &[n_conf + 1])?;
    hash_bytes.extend(bytes);
    let (conf_mean, bytes) = tensor(&file, "conf_mean", &[n_conf])?;
    hash_bytes.extend(bytes);
    let (conf_scale, bytes) = tensor(&file, "conf_scale", &[n_conf])?;
    hash_bytes.extend(bytes);
    if !conf_scale.iter().all(|s| *s > 0.0) {
        return Err(bad("conf_scale must be positive"));
    }
    let digest: String = Sha256::digest(&hash_bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    if !digest.eq_ignore_ascii_case(&file.sha256) {
        return Err(IllumError::HashMismatch {
            expected: file.sha256.clone(),
            found: digest,
        });
    }
    positive("pad_variance", file.fit.pad_variance)?;
    positive("prior_sigma", file.fit.prior_sigma)?;
    positive("temperature", hd.temperature)?;
    if !(file.fit.default_uv.iter().all(|v| v.is_finite()) && hd.bv_default.is_finite()) {
        return Err(bad("default_uv and bv_default must be finite"));
    }
    let class_prior: [f64; N_CLASSES] = hd
        .class_prior
        .clone()
        .try_into()
        .map_err(|_| bad(format!("class_prior needs {N_CLASSES} values")))?;
    if !class_prior.iter().all(|p| p.is_finite() && *p >= 0.0)
        || (class_prior.iter().sum::<f64>() - 1.0).abs() > 1e-6
    {
        return Err(bad("class_prior must be probabilities that sum to 1"));
    }
    Ok(Parsed {
        profile_id: file.profile_id,
        version: file.model_version,
        spec: HistogramSpec {
            n_bins: n,
            bin_size: hj.bin_size,
            lo_u: hj.lo_u,
            lo_v: hj.lo_v,
            min_intensity: hj.min_intensity,
            min_edge: hj.min_edge,
            channels,
            max_samples: hj.max_samples,
        },
        pad: file.fit.pad_variance,
        prior_sigma: file.fit.prior_sigma,
        default_uv: file.fit.default_uv,
        temperature: hd.temperature,
        bv_default: hd.bv_default,
        class_prior,
        hidden: hd.hidden,
        n_features: hd.n_features,
        filter,
        bias,
        w1,
        b1,
        w2,
        b2,
        feature_mean,
        feature_scale,
        conf_w,
        conf_mean,
        conf_scale,
        sha256: digest,
        param_count: file.param_count,
    })
}
