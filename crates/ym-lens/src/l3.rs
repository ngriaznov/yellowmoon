//! L3: highlight optics (`docs/21` section 4, L3). Not built yet.
//!
//! The composition is `out = (1 - a) Z + F (a_g K_g * R + a_v K_v * R + sum a_i G_i[H] + a_* B[H] + a_s ... S[H])`
//! with `a` = [`LensPlan::light_fraction`]. Until the L3 task fills in the flare terms they are zero and so is
//! `a`, which makes the pass the identity: `out = Z`.

use crate::error::LensError;
use crate::image::{ImageRef, PixelRect};
use crate::plan::LensPlan;
use crate::planes::LensPlanes;

/// Context L3 needs around a rectangle of a `fw x fh` frame at scale `k` (its core is gathered per tile).
pub(crate) fn halo(_plan: &LensPlan, _planes: &LensPlanes, _k: usize, _fw: usize, _fh: usize) -> usize {
    0
}

/// L3 on `out`: `z` is the output of L2 over `out`; `x` is the input of L1 (the flares read it).
pub(crate) fn apply(
    _plan: &LensPlan,
    _planes: &LensPlanes,
    _k: usize,
    _x: &ImageRef<'_>,
    z: Vec<f32>,
    _out: PixelRect,
) -> Result<Vec<f32>, LensError> {
    Ok(z)
}
