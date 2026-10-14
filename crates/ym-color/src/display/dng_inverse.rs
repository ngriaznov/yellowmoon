//! The DNG's scene-linear pixels for a picture whose display rendering is not ours.
//!
//! A DNG holds scene-linear AWG4 that Adobe's renderer turns into a picture with the profile fitted
//! to ACES 2 (`docs/09`, "Profile"). When C12 *is* ACES 2 the pixels are simply the C11 output
//! decoded from LogC4. When it is not, because the look carries ARRI's own display rendering
//! (LogC3 to Rec.709) or an imported ARRI display LUT is chosen, there is no scene-linear picture
//! that has that rendering in it: the DNG instead holds the scene values ACES 2 turns into the picture
//! the chain makes, so that Adobe's renderer shows approximately that picture. This module is that
//! step: display-linear Rec.709 in, scene-linear AWG4 out.
//!
//! * **The inverse of what the fitted profile emulates.** ACES 2 SDR with Rec.709 limiting
//!   primaries and the display placement (`2^`[`DISPLAY_PLACEMENT_EV`] in front of the transform,
//!   [`aces2_input_matrix`]), inverted by `Aces2Transform::inverse_ap0_linear`. The placement is in
//!   the DNG's `BaselineExposure` (`docs/09`), not in the pixels, so it is divided out here: Adobe's
//!   renderer multiplies it back in.
//! * **Display white has no finite scene value.** ACES 2 reaches its peak at 128 times the reference
//!   white, and the fitted profile's tone curve ends at 16. The result is scaled down (all three
//!   channels by one factor, so the colour keeps its ratios) until its largest channel, with the
//!   placement applied, is at most [`DNG_SCENE_LIMIT`]. A display white therefore comes out of Adobe's
//!   renderer at the display value of scene 16 (code 0.975), and the white level of the DNG stays at
//!   its floor instead of crushing the shadows.
//! * **Out-of-gamut display values clamp** (negative and above-white values, and a display LUT's
//!   wider primaries after the matrix to Rec.709): the forward transform cannot produce them.

use super::DISPLAY_PLACEMENT_EV;
use super::OutputTarget;
use super::transform::aces2_input_matrix;
use crate::aces2::{Aces2Output, Aces2Transform};
use crate::consts::Vec3;
use crate::consts::linalg::{self, Mat3};
use crate::gamut::matrices;

/// The top of the fitted DNG profile's range, as an AWG4 channel value after `BaselineExposure`
/// (placement included): `2^-fit_offset_ev` of `data/dng/profile.json`, 16 for the embedded profile.
/// Adobe's renderer clips its tone curve's input there (`docs/09`, "As built (M3-12 `tools/dngfit`)");
/// `ym-core`'s tests require it to equal the embedded profile's white floor.
pub const DNG_SCENE_LIMIT: f64 = 16.0;

/// Display-linear RGB (in the chain's output primaries) to the scene-linear AWG4 a DNG stores.
#[derive(Debug, Clone, Copy)]
pub(super) struct DngDisplayInverse {
    transform: &'static Aces2Transform,
    /// The inverse of `2^DISPLAY_PLACEMENT_EV * awg4_to_ap0` ([`aces2_input_matrix`]): AP0 to AWG4
    /// with the placement divided out.
    ap0_to_awg4: Mat3,
    /// Largest AWG4 channel the result may have: [`DNG_SCENE_LIMIT`] over `2^DISPLAY_PLACEMENT_EV`.
    max_channel: f64,
    /// The chain's output primaries to Rec.709, `None` when they are Rec.709.
    to_rec709: Option<Mat3>,
}

impl DngDisplayInverse {
    /// The inverse for a chain that renders `output`.
    pub(super) fn new(output: OutputTarget) -> Self {
        let m = matrices();
        let to_rec709 = match output {
            OutputTarget::Srgb => None,
            OutputTarget::Viewport | OutputTarget::DisplayP3 => {
                Some(linalg::mul(&m.xyz_to_rec709, &m.p3_to_xyz))
            }
            OutputTarget::Rec2020 => Some(linalg::mul(&m.xyz_to_rec709, &m.rec2020_to_xyz)),
        };
        Self {
            transform: Aces2Transform::shared(Aces2Output::Srgb),
            // `aces2_input_matrix` is `2^k` times an invertible gamut matrix: always invertible.
            ap0_to_awg4: linalg::inv(&aces2_input_matrix()).unwrap_or(m.ap0_to_awg4),
            max_channel: DNG_SCENE_LIMIT / DISPLAY_PLACEMENT_EV.exp2(),
            to_rec709,
        }
    }

    /// The scene-linear AWG4 triple for a display-linear triple (1.0 is the 100 nit white).
    pub(super) fn scene_awg4(&self, display: Vec3) -> Vec3 {
        let rec709 = match &self.to_rec709 {
            Some(m) => linalg::mul_vec(m, display),
            None => display,
        };
        let ap0 = self.transform.inverse_ap0_linear(rec709);
        let awg4 = linalg::mul_vec(&self.ap0_to_awg4, ap0);
        let top = awg4[0].max(awg4[1]).max(awg4[2]);
        if top > self.max_channel {
            let scale = self.max_channel / top;
            awg4.map(|c| c * scale)
        } else {
            awg4
        }
    }
}
