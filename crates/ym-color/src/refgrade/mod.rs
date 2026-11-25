//! The Reference Grade fit (`docs/34-reference-grade.md` sections 4b and 5, RG-06).
//!
//! A hand-written network ([`network`]) reads measures ([`measure`]) of the photo and of a reference and writes targets and gates;
//! an objective with its walls ([`objective`]) says how far a candidate picture is from them; a staged two-start search
//! ([`search`]) moves the 49 numbers of [`params`] until the objective stops falling. This is round 15 of the proof of concept,
//! written again in Rust: `f64`, a fixed order, no I/O, no threads. The per-pixel work (the chain from the node's input to the
//! display codes) is the caller's, passed in as a [`search::CandidateRenderer`].

pub mod measure;
pub mod network;
pub mod objective;
pub mod params;
pub mod search;

#[cfg(test)]
mod tests;

pub use params::{N_GLOBAL, N_LOCAL, N_PARAMS};

/// The six elements the fit grades, in the order of the node's fields.
pub const ELEMENT_NAMES: [&str; 6] = ["sky", "vegetation", "water", "ground", "built", "skin"];
/// The hue sectors: 60 degrees each, centred on 0, 60, ... 300.
pub const SECTORS: usize = 6;
/// The seed of the fit's sample (`docs/34` section 5.1).
pub const SAMPLE_SEED: u64 = 0x5EED_2026_1124;
/// The sample's grid: columns and rows, one pixel per cell.
pub const SAMPLE_GRID: (usize, usize) = (128, 64);
/// The long edge, in pixels, of the picture the fit reads.
pub const ANALYSIS_LONG_EDGE_PX: usize = 512;

/// SplitMix64: a seeded generator with no state beyond one word.
#[derive(Debug, Clone)]
pub struct SplitMix64(pub u64);

impl SplitMix64 {
    /// The next 64 bits.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1): the top 53 bits over 2^53.
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// The seeded stratified sample of `docs/34` section 5.1 over a `width x height` picture: one pixel for each cell of a 128 x 64 grid,
/// row by row, `fx = (gx + u) / 128`, `fy = (gy + u') / 64`, and the pixel `(floor(fx W), floor(fy H))` held inside the picture.
/// Returns the pixel `(x, y)` of each, the same for every candidate and depending only on the size.
///
/// # Panics
///
/// When a side is 0.
#[must_use]
pub fn sample_pixels(width: usize, height: usize) -> Vec<(usize, usize)> {
    assert!(width > 0 && height > 0, "a picture to sample");
    let mut rng = SplitMix64(SAMPLE_SEED);
    let mut out = Vec::with_capacity(SAMPLE_GRID.0 * SAMPLE_GRID.1);
    for gy in 0..SAMPLE_GRID.1 {
        for gx in 0..SAMPLE_GRID.0 {
            let fx = (gx as f64 + rng.unit()) / SAMPLE_GRID.0 as f64;
            let fy = (gy as f64 + rng.unit()) / SAMPLE_GRID.1 as f64;
            let x = ((fx * width as f64) as usize).min(width - 1);
            let y = ((fy * height as f64) as usize).min(height - 1);
            out.push((x, y));
        }
    }
    out
}
