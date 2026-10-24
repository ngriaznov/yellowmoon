//! The strength of each flare element at Flares 100: the mapping from the lens data to the light fractions L3
//! uses (`docs/21` section 4, L3, *As built (flare strength)*).
//!
//! The data give the fractions of light a lens would send into each element. They are physical for the glass
//! they describe, and for a modern coated lens they are far too faint to see: a ghost of two multi-layer
//! surfaces holds `0.3 % x 0.3 %` of the light, ten thousandths of a per cent. The top of the Flares slider is
//! therefore the lens as if its coatings were worse: the ghosts of each lens are raised until together they
//! take at least [`GHOST_FRACTION_FLOOR`] of a light's energy (the light is shared out in proportion to
//! `fraction / s`, so the small ghosts, which read as shapes, get most of it and the frame-filling ones do not
//! turn the picture milky), the attachment's likewise, and the glow and the starburst to their floors. Nothing
//! is ever lowered, a part a lens does not have stays absent, and the veil and the streak keep the data's
//! numbers.

use crate::consts::{
    ATTACHMENT_GHOST_FLOOR, GHOST_FRACTION_FLOOR, GLOW_FRACTION_FLOOR, STARBURST_FRACTION_FLOOR,
};
use crate::data::{AttachmentData, GhostData, LensData};
use crate::plan::LensPlan;

/// The fractions of the picture's light each element of L3 takes at Flares 100 (the starburst at full
/// strength, the attachment's elements at Anamorphic 100), after the mapping of [`FlareStrengths::new`].
#[derive(Clone, Debug, PartialEq)]
pub struct FlareStrengths {
    /// The glow's fraction `a_g`.
    pub glow: f64,
    /// The veiling glare's fraction `a_v`.
    pub veil: f64,
    /// The fraction of each of the lens's ghosts, in the order of its data.
    pub ghosts: Vec<f64>,
    /// The starburst's fraction at full strength.
    pub burst: f64,
    /// The streak's fraction at Anamorphic 100.
    pub streak: f64,
    /// The fraction of each of the attachment's ghosts at Anamorphic 100, in the order of its data.
    pub attachment_ghosts: Vec<f64>,
}

/// The fractions of `ghosts` raised, if their sum is below `floor`, to a sum of `floor`, shared out in
/// proportion to `fraction / s` (all of them keep at least their own fraction when the sum is already there).
fn raised_ghosts(ghosts: &[GhostData], floor: f64) -> Vec<f64> {
    let sum: f64 = ghosts.iter().map(|g| g.fraction).sum();
    if sum <= 0.0 || sum >= floor {
        return ghosts.iter().map(|g| g.fraction).collect();
    }
    let weights: Vec<f64> = ghosts
        .iter()
        .map(|g| if g.s > 0.0 { g.fraction / g.s } else { 0.0 })
        .collect();
    let total: f64 = weights.iter().sum();
    ghosts
        .iter()
        .zip(&weights)
        .map(|(g, w)| g.fraction.max(floor * w / total))
        .collect()
}

impl FlareStrengths {
    /// The strengths of `lens` behind `attachment`.
    #[must_use]
    pub fn new(lens: &LensData, attachment: &AttachmentData) -> Self {
        // A part the lens does not have stays absent: the floors only raise what is there.
        let raised = |fraction: f64, floor: f64| -> f64 {
            if fraction > 0.0 {
                fraction.max(floor)
            } else {
                0.0
            }
        };
        FlareStrengths {
            glow: raised(lens.flare.glow.fraction, GLOW_FRACTION_FLOOR),
            veil: lens.flare.veil.fraction,
            ghosts: raised_ghosts(&lens.flare.ghosts, GHOST_FRACTION_FLOOR),
            burst: raised(lens.flare.starburst.fraction, STARBURST_FRACTION_FLOOR),
            streak: attachment.streak.fraction,
            attachment_ghosts: raised_ghosts(&attachment.ghosts, ATTACHMENT_GHOST_FLOOR),
        }
    }

    /// The sum of the lens's own fractions at Flares 100 (the starburst at full strength).
    #[must_use]
    pub fn lens_fraction(&self) -> f64 {
        self.glow + self.veil + self.ghosts.iter().sum::<f64>() + self.burst
    }

    /// The sum of the attachment's fractions at Flares 100 and Anamorphic 100.
    #[must_use]
    pub fn attachment_fraction(&self) -> f64 {
        self.streak + self.attachment_ghosts.iter().sum::<f64>()
    }
}

impl LensPlan {
    /// The strengths of the plan's lens and attachment at Flares 100 ([`FlareStrengths`]).
    #[must_use]
    pub fn flare_strengths(&self) -> FlareStrengths {
        FlareStrengths::new(self.lens(), self.attachment())
    }
}
