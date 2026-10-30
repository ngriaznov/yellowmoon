//! The DNG's scene-linear pixels for a picture whose display rendering is not ours.
//!
//! A DNG holds scene-linear AWG4 that Adobe's renderer turns into a picture with the profile fitted to the
//! cine rendering (`docs/09`, "Profile"). When C12 *is* the cine rendering the pixels are simply the C11
//! output decoded from LogC4. When it is not, because the look carries ARRI's own display rendering
//! (LogC3 to Rec.709), there is no scene-linear picture that has that rendering in it: the DNG instead
//! holds the scene values the cine rendering turns into the picture the chain makes, so that Adobe's
//! renderer shows approximately that picture. This module is that step: display-linear RGB in, scene-linear
//! AWG4 out.
//!
//! * **The inverse of what the fitted profile emulates.** The cine rendering on Rec.709 primaries
//!   ([`CineRendering::inverse`]). There is no display placement to divide out: the rendering has none and
//!   the DNG carries none.
//! * **Display channels are held to `[1e-6, 1]`** by the inverse (an exact 0 has no finite pre-image; a
//!   colour the rendering cannot produce is replaced by the nearest one it can).
//! * **Display white has no finite scene value in the profile's range.** The cine rendering reaches 1.0 at
//!   +10.4 stops over grey (scene 243), and the fitted profile's tone curve ends at 16. The display colour is
//!   dimmed (all three channels by one factor, in linear light, so its hue and saturation stay) until its
//!   scene value has a largest channel of at most [`DNG_SCENE_LIMIT`]. A display white therefore comes out
//!   of Adobe's renderer at the display value of scene 16, a slightly tinted near-white keeps its tint and
//!   loses a few percent of light, and the white level of the DNG stays at its floor instead of crushing
//!   the shadows. (Scaling the scene value down instead gives a tinted near-white a strong cast, because the
//!   forward transform changes chroma with level.)

use super::OutputTarget;
use super::cine::CineRendering;
use crate::consts::Vec3;
use crate::consts::linalg::{self, Mat3};
use crate::gamut::matrices;

/// The top of the fitted DNG profile's range, as an AWG4 channel value: `2^-fit_offset_ev` of
/// `data/dng/profile.json`, 16 for the embedded profile. Adobe's renderer clips its tone curve's input there
/// (`docs/09`, "As built (M3-12 `tools/dngfit`)"); `ym-core`'s tests require it to equal the embedded
/// profile's white floor.
pub const DNG_SCENE_LIMIT: f64 = 16.0;

/// Display-linear RGB (in the chain's output primaries) to the scene-linear AWG4 a DNG stores.
#[derive(Debug, Clone, Copy)]
pub(super) struct DngDisplayInverse {
    /// The rendering the fitted profile emulates: the cine rendering on Rec.709 primaries.
    rendering: CineRendering,
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
            rendering: CineRendering::new(OutputTarget::Srgb),
            to_rec709,
        }
    }

    /// The scene-linear AWG4 triple that the cine rendering turns into `rec709` (display-linear Rec.709).
    /// Not limited.
    fn unlimited(&self, rec709: Vec3) -> Vec3 {
        self.rendering.inverse(rec709).scene.0
    }

    /// The scene-linear AWG4 triple for a display-linear triple (1.0 is the 100 nit white).
    ///
    /// A display colour near white has a scene value above the top of the DNG profile
    /// ([`DNG_SCENE_LIMIT`]). It is made to fit by **dimming the display colour** (all three channels by one
    /// factor, in linear light) until its scene value fits, and not by scaling the scene value down, which
    /// would turn a slightly tinted highlight into a strong cast.
    pub(super) fn scene_awg4(&self, display: Vec3) -> Vec3 {
        let rec709 = match &self.to_rec709 {
            Some(m) => linalg::mul_vec(m, display),
            None => display,
        };
        // The display's range is `0..=1` (the 100 nit peak of SDR); not a number is 0.
        let rec709 = rec709.map(|c| if c.is_nan() { 0.0 } else { c.clamp(0.0, 1.0) });
        let top = |a: Vec3| a[0].max(a[1]).max(a[2]);
        let awg4 = self.unlimited(rec709);
        if top(awg4) <= DNG_SCENE_LIMIT {
            return awg4;
        }
        // The largest factor in 0.5..=1 whose colour fits (the scene value grows with the factor; 0.5 of
        // any display colour is far below the limit). `DIM_STEPS` halvings leave 3e-5.
        let (mut fits, mut over) = (0.5_f64, 1.0_f64);
        for _ in 0..DIM_STEPS {
            let mid = 0.5 * (fits + over);
            if top(self.unlimited(rec709.map(|c| c * mid))) <= DNG_SCENE_LIMIT {
                fits = mid;
            } else {
                over = mid;
            }
        }
        let awg4 = self.unlimited(rec709.map(|c| c * fits));
        // The backstop for a colour that does not fit even at half: scale the scene value.
        let top = top(awg4);
        if top > DNG_SCENE_LIMIT {
            awg4.map(|c| c * (DNG_SCENE_LIMIT / top))
        } else {
            awg4
        }
    }
}

/// Halvings of the bracket `0.5..=1` of the display dimming factor in [`DngDisplayInverse::scene_awg4`].
const DIM_STEPS: u32 = 14;
