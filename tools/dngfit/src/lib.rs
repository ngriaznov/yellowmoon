//! `dngfit`: fits the embedded DNG profile of the baked-look export and checks the DNG gates (`docs/09-export.md`,
//! section DNG with the look baked in).
//!
//! * [`adobe`] is Adobe's reference renderer (`dng_validate`) as a function from scene-linear AWG4 colours to sRGB.
//! * [`target`] is what Yellowmoon shows: the ACES 2 output transform for sRGB.
//! * [`profile_file`] is `data/dng/profile.json`, the fitted profile `ym-export` embeds.

#![warn(missing_docs)]

pub mod adobe;
pub mod check;
pub mod fit;
pub mod lookfit;
pub mod model;
pub mod profile_file;
pub mod samples;
pub mod target;
