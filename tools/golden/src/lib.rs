//! Golden-image tooling (`docs/13-testing.md`, section Golden images).
//!
//! * [`synth`] turns the synthetic scenes of `training/synth` (raw mosaics with JSON sidecars) into ARW files with
//!   `ym_raw::testgen`, one per variant, and checks that decoding each file gives the mosaic back.
//! * [`scene`] reads those scenes.
//! * The render and compare runner (CLI renders against references, CIEDE2000) is a later task; `main.rs` keeps its stub.

#![warn(missing_docs)]

pub mod demosaic;
pub mod preview;
pub mod scene;
pub mod synth;
