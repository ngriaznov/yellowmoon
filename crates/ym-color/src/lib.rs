//! Colour pipeline maths: the CPU reference for every colour stage.
//!
//! Pure maths with no GPU dependency: LogC3/LogC4, AWG3/AWG4, ARRI's CCT matrices, the sensor map,
//! white balance, CDL, cube LUTs, the ACES 2 output transform, the illuminant estimator and the camera
//! default rendering of the Before view. Colour maths runs in f64 here; the WGSL kernels in `ym-gpu` are
//! checked against these functions.
//!
//! Specs: `docs/02-colour-pipeline.md` (stages C1 to C13, constants and tests) and
//! `docs/08-looks.md` (look application).

#![warn(missing_docs)]

pub mod aces2;
pub mod alf;
pub mod arri;
pub mod camera_default;
pub mod cdl;
pub mod chain;
pub mod consts;
pub mod cube;
pub mod curve;
pub mod deltae;
pub mod display;
pub mod encode;
pub mod exposure;
pub mod gamut;
pub mod grade;
pub mod illum;
pub mod logc;
pub mod look;
pub mod lut;
pub mod sensor_map;
pub mod spectral;
pub mod tone;
pub mod warp;
pub mod wb;
