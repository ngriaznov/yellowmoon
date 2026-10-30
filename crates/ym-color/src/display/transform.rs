//! Stage C12: the display transform, the cine rendering (`docs/02-colour-pipeline.md` section C12).
//!
//! C12 takes the LogC4 output of C11 and returns **display-linear** RGB in the primaries of the chosen
//! [`OutputTarget`](super::OutputTarget), with 1.0 the 100 nit SDR white (the value C13 then encodes, and
//! the value an exporter resizes in). LogC4 is decoded to linear AWG4 and handed to
//! [`CineRendering::forward`]: there is no step in front of it (no white lift, no red steps, no display
//! placement), and every output uses the gamut of its own primaries.

use super::OutputTarget;
use super::cine::CineRendering;
use crate::consts::Vec3;
use crate::logc::LogC4;

/// C12 prepared for one output.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PreparedDisplay {
    cine: CineRendering,
}

impl PreparedDisplay {
    /// Prepares C12 for `output`.
    pub(crate) fn new(output: OutputTarget) -> Self {
        Self {
            cine: CineRendering::new(output),
        }
    }

    /// C12: LogC4 to display-linear RGB in the output's primaries (1.0 is 100 nit white).
    pub(crate) fn display_linear(&self, v: LogC4) -> Vec3 {
        self.cine.forward(v.to_linear())
    }
}
