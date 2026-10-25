//! The colour of the light the field stages move (`docs/21` section 4, *As built (the light's colour)*).
//!
//! The lens plane is camera RGB balanced with `g0`: channel `c` clips at `g0_c`, and `g0` is 1 for green and
//! 1.5 to 3 for red and blue, so green clips first for any white light. Between green's clip level and the
//! others' a white light reads magenta (green is a lower bound, the others measure), and R5, which rebuilds the
//! clipped channel from a calibration on the fringe around the light, stays under it. A lens stage that spreads
//! such a pixel spreads the cast with it, and where the spread falls under display white the cast shows: a white
//! star turned violet.
//!
//! [`lit_input`] gives L1 and L2 the input they should spread: the picture with every pixel's light in its own
//! colour ([`Restore::light_colour`](crate::l3::Restore), the rule the flares' restored light uses too, so all
//! the stages agree). It is a function of one pixel, so a tile and the whole frame give the same bits. The
//! picture itself is not changed: L3 reads the picture as it is and the lens-off path never comes here.

use rayon::prelude::*;

use crate::consts::{CLIP_RAMP, LIGHT_SATURATION_RAMP};
use crate::image::ImageRef;
use crate::l3::Restore;
use crate::plan::LensPlan;

/// What the device pass that gives L1 and L2 the light's colour takes of a plan: the clip levels and the clip
/// test's ramp (`LensPlan::light_pass`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LightPass {
    /// `g0` of each channel (0: the channel has no clip level).
    pub clip: [f32; 3],
    /// The clip test's ramp for the picture's own pixels: from, to.
    pub ramp: (f32, f32),
}

impl LensPlan {
    /// The device pass that gives the stages that move light (L1 and L2) the light's colour, or `None` when
    /// neither acts (Edges and Anamorphic both 0: `Z` is the picture itself and nothing is spread).
    #[must_use]
    pub fn light_pass(&self) -> Option<LightPass> {
        if self.field_is_identity() {
            return None;
        }
        Some(LightPass {
            clip: self.frame().clip,
            ramp: (CLIP_RAMP.0 as f32, CLIP_RAMP.1 as f32),
        })
    }
}

/// The input of L1 and L2 with the colour of the light it holds, or `None` when no pixel changes (no channel
/// is saturated anywhere, or the plan has no field stage: Edges and Anamorphic both 0, where `Z` is the
/// picture itself). The CPU reference of the device pass `lens_light`.
///
/// The result has the samples of `input` (planar, `3 * width * height`).
#[must_use]
pub fn lit_input(plan: &LensPlan, input: &ImageRef) -> Option<Vec<f32>> {
    if plan.field_is_identity() {
        return None;
    }
    let clip = plan.frame().clip;
    let n = input.width * input.height;
    // Nothing is saturated: every channel under the ramp's start, in its own clip levels.
    let from = LIGHT_SATURATION_RAMP.0;
    let below = |c: usize| {
        let g0 = f64::from(clip[c]);
        g0 <= 0.0
            || input.data[c * n..(c + 1) * n]
                .iter()
                .all(|v| f64::from(*v) < from * g0)
    };
    if below(0) && below(1) && below(2) {
        return None;
    }
    let rs = Restore::new(plan, CLIP_RAMP);
    let mut out = input.data.to_vec();
    let (r, rest) = out.split_at_mut(n);
    let (g, b) = rest.split_at_mut(n);
    r.par_chunks_mut(input.width)
        .zip(g.par_chunks_mut(input.width))
        .zip(b.par_chunks_mut(input.width))
        .for_each(|((r, g), b)| {
            for ((r, g), b) in r.iter_mut().zip(g.iter_mut()).zip(b.iter_mut()) {
                let x = [f64::from(*r), f64::from(*g), f64::from(*b)];
                let y = rs.light_colour(x);
                if y != x {
                    *r = y[0] as f32;
                    *g = y[1] as f32;
                    *b = y[2] as f32;
                }
            }
        });
    Some(out)
}
