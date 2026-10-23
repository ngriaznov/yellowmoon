//! Learned-model runtime: loads the shipped weights and runs them on tiles, and the crop student
//! of Auto crop ([`crop_net`]).
//!
//! Backends sit behind one `TileModel` trait (burn on the shared wgpu device, or ONNX Runtime;
//! chosen by the M0 benchmark), with a CPU fallback.
//!
//! Specs: `docs/07-ml-models.md` (models, architecture, runtime, parity and versioning, gates).

#![warn(missing_docs)]

pub mod backends;
pub mod cpu_runtime;
pub mod crop_net;
pub mod model;
pub mod tiles;
