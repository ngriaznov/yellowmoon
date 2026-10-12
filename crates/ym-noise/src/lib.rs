//! Noise estimation for every camera: a per-image Poisson-Gaussian model and the sigma map the
//! denoise network consumes.
//!
//! Nothing is stored between sessions; priors are fitted from public files and shipped as data.
//!
//! Specs: `docs/06-noise.md`; background in `research/notes/noise-estimation.md`.

#![warn(missing_docs)]

pub mod estimate;
pub mod model;
pub mod priors;
pub mod sigma_map;
