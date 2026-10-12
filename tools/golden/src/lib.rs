//! Golden-image tooling (`docs/13-testing.md`, section Golden images).
//!
//! * [`runner`] renders the golden files through the real `ym` binary (`ym render --json --settings ...`) and compares
//!   the pictures with the stored ones: `golden check` and `golden update`. [`manifest`], [`expected`], [`image`] and
//!   [`ymrun`] are its parts.
//! * [`truth`] develops the synthetic scenes and compares the colour with their ground truth, patch by patch
//!   (`golden truth`); [`dngread`] reads the DNG it develops.
//! * [`synth`] turns the synthetic scenes of `training/synth` (raw mosaics with JSON sidecars) into ARW files with
//!   `ym_raw::testgen`, one per variant, and checks that decoding each file gives the mosaic back. [`scene`] reads
//!   those scenes.

#![warn(missing_docs)]

pub mod demosaic;
pub mod dngread;
pub mod expected;
pub mod image;
pub mod manifest;
pub mod preview;
pub mod runner;
pub mod scene;
pub mod synth;
pub mod truth;
pub mod ymrun;
