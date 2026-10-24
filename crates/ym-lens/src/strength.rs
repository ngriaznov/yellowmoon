//! The strength of each flare element at Flares 100: the mapping from the lens data to the light fractions L3
//! uses (`docs/21` section 4, L3, *As built (flare strength)*).
//!
//! The data give the fractions of light a lens would send into each element. They are physical for the glass
//! they describe, and for a modern coated lens they are far too faint to see: a ghost of two multi-layer
//! surfaces holds `0.3 % x 0.3 %` of the light, ten thousandths of a per cent. The top of the Flares slider is
//! therefore the lens as if its coatings were worse: each lens's ghosts are raised to the level of single-layer
//! or bare glass (`GHOST_FRACTION_FLOOR`), never lowered, and the attachment's likewise. Glow, veil and the
//! starburst keep the data's numbers; the streak's fraction is data (`anamorphic.json`).

use crate::consts::{
    ATTACHMENT_GHOST_FLOOR, GHOST_FRACTION_FLOOR, GHOST_GAIN_MAX, GLOW_FRACTION_FLOOR,
    STARBURST_FRACTION_FLOOR,
};
use crate::data::{AttachmentData, LensData};
use crate::plan::LensPlan;

/// The fractions of the picture's light each element of L3 takes at Flares 100 (the starburst at full
/// strength, the attachment's elements at Anamorphic 100), after the mapping of [`FlareStrengths::new`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlareStrengths {
    /// The glow's fraction `a_g`.
    pub glow: f64,
    /// The veiling glare's fraction `a_v`.
    pub veil: f64,
    /// The factor every ghost's fraction is multiplied by (at least 1).
    pub ghost_gain: f64,
    /// The starburst's fraction at full strength.
    pub burst: f64,
    /// The streak's fraction at Anamorphic 100.
    pub streak: f64,
    /// The factor the attachment's ghosts' fractions are multiplied by (at least 1).
    pub attachment_ghost_gain: f64,
}

impl FlareStrengths {
    /// The strengths of `lens` behind `attachment`.
    #[must_use]
    pub fn new(lens: &LensData, attachment: &AttachmentData) -> Self {
        // A part the lens does not have stays absent: the floors only raise what is there.
        let gain = |sum: f64, floor: f64| -> f64 {
            if sum > 0.0 {
                (floor / sum).clamp(1.0, GHOST_GAIN_MAX)
            } else {
                1.0
            }
        };
        let raised = |fraction: f64, floor: f64| -> f64 {
            if fraction > 0.0 {
                fraction.max(floor)
            } else {
                0.0
            }
        };
        let ghosts: f64 = lens.flare.ghosts.iter().map(|g| g.fraction).sum();
        let att_ghosts: f64 = attachment.ghosts.iter().map(|g| g.fraction).sum();
        FlareStrengths {
            glow: raised(lens.flare.glow.fraction, GLOW_FRACTION_FLOOR),
            veil: lens.flare.veil.fraction,
            ghost_gain: gain(ghosts, GHOST_FRACTION_FLOOR),
            burst: raised(lens.flare.starburst.fraction, STARBURST_FRACTION_FLOOR),
            streak: attachment.streak.fraction,
            attachment_ghost_gain: gain(att_ghosts, ATTACHMENT_GHOST_FLOOR),
        }
    }

    /// The sum of the lens's own fractions at Flares 100 (the starburst at full strength).
    #[must_use]
    pub fn lens_fraction(&self, lens: &LensData) -> f64 {
        self.glow
            + self.veil
            + self.ghost_gain * lens.flare.ghosts.iter().map(|g| g.fraction).sum::<f64>()
            + self.burst
    }

    /// The sum of the attachment's fractions at Flares 100 and Anamorphic 100.
    #[must_use]
    pub fn attachment_fraction(&self, attachment: &AttachmentData) -> f64 {
        self.streak
            + self.attachment_ghost_gain * attachment.ghosts.iter().map(|g| g.fraction).sum::<f64>()
    }
}

impl LensPlan {
    /// The strengths of the plan's lens and attachment at Flares 100 ([`FlareStrengths`]).
    #[must_use]
    pub fn flare_strengths(&self) -> FlareStrengths {
        FlareStrengths::new(self.lens(), self.attachment())
    }
}
