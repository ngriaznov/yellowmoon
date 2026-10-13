//! GPU image processing: wgpu device and queue ownership, tiling with halos, buffer pools and
//! the WGSL compute kernels.
//!
//! Every kernel has a CPU reference in [`cpu`] and a parity test with a stated maximum absolute
//! error. Depends on `ym-color` (constants and reference functions used in parity tests) and `ym-noise` (the noise model the raw algorithms take on the 16-bit working scale, `cpu::NoiseDn`).
//! WGSL sources live in `src/shaders/`.
//!
//! Specs: `docs/01-architecture.md` sections GPU tiling and Scheduling and threads,
//! `docs/05-raw-processing.md` (stages R1 to R9, tiling and halos) and `docs/02-colour-pipeline.md`.

#![warn(missing_docs)]

pub mod cpu;
pub mod device;
pub mod kernels;
pub mod pool;
pub mod queue;
pub mod tiles;
