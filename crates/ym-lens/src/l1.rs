//! L1: bokeh reshaping (`docs/21` section 4, L1). Not built yet.
//!
//! The stage is the identity until the L1 task fills it in. The module keeps the shape of the composition
//! `Z = L2(L1(X))` so that the public API does not change when it does: [`apply`] returns the picture L2 reads
//! over a rectangle, and [`halo`] the context L1 itself needs around that rectangle.

use crate::error::LensError;
use crate::image::{ImageRef, PixelRect};
use crate::plan::LensPlan;
use crate::planes::LensPlanes;

/// A picture a stage hands to the next: the caller's input, borrowed, or a new buffer.
pub(crate) enum Stage<'a> {
    /// The stage changed nothing.
    Same(ImageRef<'a>),
}

impl<'a> Stage<'a> {
    /// The picture.
    pub(crate) fn image(&self) -> ImageRef<'a> {
        match self {
            Stage::Same(i) => *i,
        }
    }
}

/// Context L1 needs around a rectangle of a `fw x fh` frame at scale `k`.
pub(crate) fn halo(
    _plan: &LensPlan,
    _planes: &LensPlanes,
    _k: usize,
    _fw: usize,
    _fh: usize,
) -> usize {
    0
}

/// L1 over `want` (L2's receptive field). `input` covers `want` grown by [`halo`].
pub(crate) fn apply<'a>(
    _plan: &LensPlan,
    _planes: &LensPlanes,
    _k: usize,
    input: &ImageRef<'a>,
    _want: PixelRect,
) -> Result<Stage<'a>, LensError> {
    Ok(Stage::Same(*input))
}
