//! Spectral engine: grid tables with mired interpolation, `w_A`, the camera locus, camera RGB to CCT and
//! Duv and back, and a loader for rawtoaces-format curve files (`docs/02-colour-pipeline.md` sections
//! C3-C4 and "Camera RGB <-> CCT and Duv"; the reading rules are in `training/sensor_map/README.md`).
//!
//! The fit scripts precompute everything that needs CIE data (the blackbody and daylight `w_A` and camera
//! locus, the class reference lights) into `profile.json`; this module only reads those tables:
//!
//! | Piece | Type | What it does |
//! |---|---|---|
//! | [`grid`] | [`Grid`] | `{ start_k, step_k, values }` with mired weights, linear or geometric blending |
//! | [`white`] | [`AlexaWhite`] | `w_A` for blackbody/daylight and the fluorescent/LED class lists, blended by class probability |
//! | [`locus`] | [`Locus`] | camera log-chroma and xy against CCT; camera RGB to (CCT, Duv) and back |
//! | [`light`] | [`LightClass`], [`ClassProbabilities`] | the four profile classes |
//! | [`curves`] | [`SpectralCurves`] | rawtoaces-data JSON loader, for tests and tools |
//! | [`fixtures`] | `Fixtures` | reader for the git-ignored `testdata/spectral/` files of the Python exporter, for tests and tools |
//!
//! The scene-light rule everywhere is the rawtoaces convention: blackbody below
//! [`KELVIN_SPLIT_K`] (4000 K), CIE daylight from there. Nothing here depends on CIE datasets, so none
//! is compiled into the app.
//!
//! # Accuracy
//!
//! The arithmetic is the same as `runtime_ref.py`, so the Rust values equal the interpolation of the
//! shipped tables to 1e-9 (`tests/spectral.rs` against `tests/data/spectral_reference.json`). Against
//! the exact spectra, off the 10 K grid, the tables are good to the tolerances recorded in that file
//! (3e-5 for blackbody `w_A` below 2000 K, 5e-6 above; 4e-4 for CIE daylight, whose D series as built
//! by colour-science is a staircase in temperature).

pub mod curves;
pub mod fixtures;
pub mod grid;
pub mod light;
pub mod locus;
pub mod white;

pub use curves::{CurveError, SpectralCurves};
pub use grid::{Grid, GridError, GridValue, bracket_by_mired};
pub use light::{ClassProbabilities, ClassProbabilityError, LightClass};
pub use locus::{CctDuv, DuvScale, Locus, LocusXy};
pub use white::{AlexaWhite, WhiteEntry};

/// Blackbody tables serve below this temperature, CIE daylight tables from it (kelvin).
pub const KELVIN_SPLIT_K: f64 = 4000.0;

/// First temperature of the blackbody tables (kelvin).
pub const BLACKBODY_START_K: f64 = 1500.0;

/// First temperature of the CIE daylight tables (kelvin); the D series is only defined from here.
pub const DAYLIGHT_START_K: f64 = 4000.0;

/// Last temperature of the daylight tables and of the blackbody `w_A` table (kelvin); the blackbody
/// locus ends at [`KELVIN_SPLIT_K`].
pub const TABLE_END_K: f64 = 25000.0;

/// Spacing of the `w_A`, log-chroma and xy tables (kelvin).
pub const GRID_STEP_K: f64 = 10.0;

/// Spacing of the Duv scale tables (kelvin).
pub const DUV_SCALE_STEP_K: f64 = 100.0;

/// Errors from reading or validating spectral tables.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum SpectralError {
    /// A table does not have the structure the profile format promises.
    #[error("{table}: {reason}")]
    InvalidTable {
        /// Path of the table inside the profile, for example `locus.daylight`.
        table: String,
        /// What is wrong.
        reason: String,
    },
    /// A camera RGB for the CCT readout must have three positive finite channels.
    #[error("camera RGB {0:?} must have three positive finite channels")]
    InvalidRgb([f64; 3]),
    /// Both locus branches have only zero-length chords (cannot happen for a validated profile).
    #[error("the locus has no chord with a direction")]
    DegenerateLocus,
}

/// Checks that a grid starts at `start_k`, has step `step_k` and ends at `end_k`.
///
/// # Errors
///
/// [`SpectralError::InvalidTable`] naming `table` when any of the three differs.
pub(crate) fn check_grid_range<T: GridValue>(
    table: &str,
    grid: &Grid<T>,
    start_k: f64,
    end_k: f64,
    step_k: f64,
) -> Result<(), SpectralError> {
    let ok = grid.start_k() == start_k && grid.step_k() == step_k && grid.end_k() == end_k;
    if ok {
        Ok(())
    } else {
        Err(SpectralError::InvalidTable {
            table: table.to_string(),
            reason: format!(
                "expected {start_k} to {end_k} K every {step_k} K, found {} to {} K every {} K ({} values)",
                grid.start_k(),
                grid.end_k(),
                grid.step_k(),
                grid.len()
            ),
        })
    }
}
