//! The MediaPipe face networks (Apache-2.0, Google): BlazeFace full range, Face Mesh V2 and Selfie Multiclass (`docs/27`
//! S-08 to S-10, `docs/07`).
//!
//! The networks were converted from TFLite to safetensors by `tools/mediapipe/convert.py` (never run by the app) and run
//! here by [`graph`], a runner for the operators they use. The model folder is `data/models/mediapipe/`, fetched by
//! `tools/fetch/mediapipe.sh`. S-08 is the face finder, [`blazeface`]; the mesh and the segmenter come with S-09 and S-10.

use std::path::Path;

use crate::model::{MlError, Result, verify_sha256};

pub mod blazeface;
pub mod graph;

/// The line for `file` in `dir/SHA256SUMS`.
pub(crate) fn expected_sha256(dir: &Path, file: &str) -> Result<String> {
    let sums = std::fs::read_to_string(dir.join("SHA256SUMS"))?;
    sums.lines()
        .filter_map(|l| l.split_once("  "))
        .find(|(_, name)| name.trim() == file)
        .map(|(hash, _)| hash.trim().to_owned())
        .ok_or_else(|| MlError::Weights(format!("SHA256SUMS has no line for {file}")))
}

pub(crate) fn verify(what: &str, bytes: &[u8], want: &str) -> Result<()> {
    verify_sha256(what, bytes, want)
}
