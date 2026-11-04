//! `ym-film`: the Film node's engine, the per-pixel colour reference (`docs/24-film.md` section 3).
//!
//! A colour photo goes through a simulated colour negative, an enlarger, a print and a scan. Every stage is a
//! function of its own and can be tested alone; [`FilmChain`] strings them together:
//!
//! | # | Stage | Module |
//! |---|---|---|
//! | 1 | scene light | [`scene`] |
//! | 2 | film exposure | [`exposure`], [`spectra`] |
//! | 4 | characteristic curves | [`develop`] |
//! | 5 | DIR couplers | [`develop`] |
//! | 7 | enlarger | [`enlarger`] |
//! | 8 | paper curves | [`enlarger`] |
//! | 9 | scan | [`scan`] |
//! | 10 | return to LogC4 / AWG4 | [`chain`] |
//!
//! Stages 3 (scatter and halation) and 6 (grain) and the diffusion of stage 5 are spatial and belong to F-05.
//! Baking into tables is F-04. Everything here is f64, deterministic, and has no `fma`.
//!
//! The crate is written from the published science, not from any film-simulation code: see the header of each
//! module for the papers and data sheets it follows (`docs/24` section 3.1).

#![warn(missing_docs)]

pub mod chain;
pub mod cie;
pub mod data;
pub mod develop;
pub mod enlarger;
pub mod error;
pub mod exposure;
pub mod math;
pub mod params;
pub mod profile;
pub mod scan;
pub mod scene;
pub mod spectra;

pub use chain::{FilmChain, PaperChoice, Taps};
pub use data::{DataDirError, find_data_dir};
pub use error::FilmError;
pub use params::{FilmParams, Output};
pub use profile::{Names, Profile, StockKind, StockName};
pub use spectra::{ReferenceSpectra, Spectra, TableSpectra};

/// Number of wavelength samples: 380 to 780 nm in 5 nm steps.
pub const N_LAMBDA: usize = 81;
/// First wavelength of the grid, in nm.
pub const LAMBDA_START_NM: f64 = 380.0;
/// Step of the grid, in nm.
pub const LAMBDA_STEP_NM: f64 = 5.0;

/// The wavelength of grid sample `i`, in nm.
#[must_use]
pub fn lambda_nm(i: usize) -> f64 {
    LAMBDA_START_NM + LAMBDA_STEP_NM * i as f64
}

/// Three values (a colour, three densities, three exposures).
pub type Vec3 = [f64; 3];
/// One sample per wavelength of the grid.
pub type Spectrum = [f64; N_LAMBDA];
