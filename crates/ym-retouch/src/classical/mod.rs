//! Classical (no learned weights) stand-ins for the two models of `docs/27` section 4 that blemish
//! removal needs: [`ClassicalProposer`] (a [`crate::BlemishProposer`]) and [`ClassicalBeautyMarks`]
//! (a [`crate::BeautyMarkDetector`]). Both are deterministic, `f64`, with rows in parallel.

mod grey;
mod moles;
mod proposer;
#[cfg(test)]
mod protocol;
#[cfg(test)]
mod synth;
#[cfg(test)]
mod tests;

pub use moles::{ClassicalBeautyMarks, brown_ratio};
pub use proposer::{ClassicalProposer, spot_score};

/// The constants of the classical models. All are starting values for S-12 (tuning).
pub mod constants {
    pub use super::moles::{
        BOX_PAD, DETECT_MAX_SIDE_PX, DOG_MIN, DOG_RATIO, DOG_SIGMA_FRACS, MAX_ASPECT,
        MAX_CANDIDATES, MAX_CONTRAST, MAX_MARKS, MEASURE_SIGMA_PX, MIN_BROWN_RATIO, MIN_CONTRAST,
        MIN_FILL, MIN_RED_DROP, MOLE_MAX_FRAC, MOLE_MIN_FRAC, SIZE_TOLERANCE,
    };
    pub use super::proposer::{
        ASPECT_MAX, DARK_HI, DETAIL_CLAMP, DILATE_FRAC, FILL_SWEEPS, LO_RATIO, LOG_FLOOR,
        MAX_SIDE_FRAC, MIN_SPOT_SIDE_PX, PRE_SIGMA_FRAC, PRE_SIGMA_MIN_PX, RED_HI, SEP_SIGMA_FRAC,
        SEP_SIGMA_MIN_PX, SOR_OMEGA, SPOT_MAX_FRAC, SPOT_MIN_FRAC, TILE_PX,
    };
}
