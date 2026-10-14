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
//!   [`aces2_input_matrix`](super::aces2_input_matrix)), inverted by `Aces2Transform::inverse_ap0_linear`. The placement is in
//!   the DNG's `BaselineExposure` (`docs/09`), not in the pixels, so it is divided out here: Adobe's
//!   renderer multiplies it back in.
//! * **Display white has no finite scene value.** ACES 2 reaches its peak at 128 times the reference
//!   white, and the fitted profile's tone curve ends at 16. The display colour is dimmed (all three
//!   channels by one factor, in linear light, so its hue and saturation stay) until its scene value, with
//!   the placement applied, has a largest channel of at most [`DNG_SCENE_LIMIT`]. A display white
//!   therefore comes out of Adobe's renderer at the display value of scene 16 (linear 0.9435, code
//!   0.975), a slightly tinted near-white keeps its tint and loses a few percent of light, and the
//!   white level of the DNG stays at its floor instead of crushing the shadows. (Scaling the scene
//!   value down instead, as this module first did, gave a tinted near-white a strong cast: the forward
//!   transform compresses chroma by lightness, so the same scene ratios at a lower lightness are far
//!   more saturated. `(1, 1, 0.95)` came out as `(1, 0.99, 0.63)`.)
//! * **Out-of-gamut display values clamp** (negative and above-white values, and a display LUT's
//!   wider primaries after the matrix to Rec.709): the forward transform cannot produce them.

use super::DISPLAY_PLACEMENT_EV;
use super::OutputTarget;
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
    /// The inverse of `2^DISPLAY_PLACEMENT_EV * awg4_to_ap0` ([`aces2_input_matrix`](super::aces2_input_matrix)): AP0 to AWG4
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
        let gain = DISPLAY_PLACEMENT_EV.exp2();
        let to_rec709 = match output {
            OutputTarget::Srgb => None,
            OutputTarget::Viewport | OutputTarget::DisplayP3 => {
                Some(linalg::mul(&m.xyz_to_rec709, &m.p3_to_xyz))
            }
            OutputTarget::Rec2020 => Some(linalg::mul(&m.xyz_to_rec709, &m.rec2020_to_xyz)),
        };
        Self {
            transform: Aces2Transform::shared(Aces2Output::Srgb),
            // The inverse of `aces2_input_matrix()` (`2^k` times AWG4 to AP0) is AP0 to AWG4 over `2^k`.
            ap0_to_awg4: m.ap0_to_awg4.map(|row| row.map(|v| v / gain)),
            max_channel: DNG_SCENE_LIMIT / gain,
            to_rec709,
        }
    }

    /// The scene-linear AWG4 triple, with the placement divided out, that ACES 2 turns into `rec709`
    /// (display-linear Rec.709, already cleaned to `0..=1`). Not limited.
    fn unlimited(&self, rec709: Vec3) -> Vec3 {
        linalg::mul_vec(&self.ap0_to_awg4, self.transform.inverse_ap0_linear(rec709))
    }

    /// The scene-linear AWG4 triple for a display-linear triple (1.0 is the 100 nit white).
    ///
    /// A display colour near white has a scene value above the top of the DNG profile
    /// ([`DNG_SCENE_LIMIT`]). It is made to fit by **dimming the display colour** (all three channels by one
    /// factor, in linear light) until its scene value fits, and not by scaling the scene value down: the
    /// forward transform compresses chroma more the lighter the colour, so a scene value brought down to
    /// the limit with its ratios kept shows at a lower lightness with far less compression and turns a
    /// slightly tinted highlight, `(1, 1, 0.95)` for example, into a strong cast (ΔE76 20 in the test
    /// colours), where the dimmed colour keeps its hue and its saturation and loses a few percent of
    /// light (ΔE76 at most 2.3, the loss of a white).
    pub(super) fn scene_awg4(&self, display: Vec3) -> Vec3 {
        let rec709 = match &self.to_rec709 {
            Some(m) => linalg::mul_vec(m, display),
            None => display,
        };
        // What the inverse does first (`Aces2Transform::inverse_ap0_linear`): not a number is 0, and the
        // display's range is `0..=1` (the 100 nit peak of SDR).
        let rec709 = rec709.map(|c| if c.is_nan() { 0.0 } else { c.clamp(0.0, 1.0) });
        let top = |a: Vec3| a[0].max(a[1]).max(a[2]);
        let awg4 = self.unlimited(rec709);
        if top(awg4) <= self.max_channel {
            return awg4;
        }
        // The largest factor in 0.5..=1 whose colour fits (the scene value grows with the factor; 0.5 of
        // any display colour is far below the limit). `DIM_STEPS` halvings leave 3e-5.
        let (mut fits, mut over) = (0.5_f64, 1.0_f64);
        for _ in 0..DIM_STEPS {
            let mid = 0.5 * (fits + over);
            if top(self.unlimited(rec709.map(|c| c * mid))) <= self.max_channel {
                fits = mid;
            } else {
                over = mid;
            }
        }
        let awg4 = self.unlimited(rec709.map(|c| c * fits));
        // The backstop for a colour that does not fit even at half: scale the scene value.
        let top = top(awg4);
        if top > self.max_channel {
            awg4.map(|c| c * (self.max_channel / top))
        } else {
            awg4
        }
    }
}

/// Halvings of the bracket `0.5..=1` of the display dimming factor in [`DngDisplayInverse::scene_awg4`].
const DIM_STEPS: u32 = 14;
