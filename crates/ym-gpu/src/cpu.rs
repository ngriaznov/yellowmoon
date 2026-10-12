//! CPU reference implementations of every kernel, used for parity tests and for machines without a usable GPU (`docs/05-raw-processing.md`).

pub mod demosaic;
pub mod denoise;
pub mod detail;
pub mod gains;
pub mod highlights;
pub mod lens;
pub mod normalize;
pub mod resample;
pub mod sharpen;
pub mod vignette;
