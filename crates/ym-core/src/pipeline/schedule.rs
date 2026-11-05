//! The scheduler's registry of kinds (`docs/23` section 5.12).
//!
//! Every kind of node and every kind of mask row says here which footprint it has: per pixel or spatial, how far it
//! reads and what it reads (`ym_color::schedule`). Both functions are **exhaustive `match`es with no wildcard arm**:
//! adding a variant to [`NodeKind`] or [`MaskKind`] does not compile until the new kind is given its step kind, which
//! in turn has to declare its footprint (`StepKind::declared`, `OpKind::declared` in `ym-color`). The scheduler itself
//! (`ColorChain::schedule`, `Schedule::infer`) never looks at a kind, only at the footprints.
//!
//! The random-graph test (`tests/any_graph.rs`) draws its kinds from [`NodeKind::ALL`] and [`MaskKind::ALL`] and fails
//! when a kind that is not marked as excluded there never appeared in a run.

use ym_color::mask::OpKind;
use ym_color::schedule::{Declared, StepKind};

use crate::settings::{MaskKind, NodeKind};

/// The plan step a node kind runs as, or `None` for the Input and the Output, which are not plan steps.
#[must_use]
pub fn step_kind(kind: NodeKind) -> Option<StepKind> {
    match kind {
        NodeKind::Input | NodeKind::Output => None,
        NodeKind::Corrector => Some(StepKind::Corrector),
        NodeKind::Film => Some(StepKind::Film),
    }
}

/// The prepared row kind a mask kind runs as.
#[must_use]
pub fn op_kind(kind: MaskKind) -> OpKind {
    match kind {
        MaskKind::Object
        | MaskKind::Subject
        | MaskKind::People
        | MaskKind::Background
        | MaskKind::Brush => OpKind::Raster,
        MaskKind::Linear => OpKind::Linear,
        MaskKind::Radial => OpKind::Radial,
        MaskKind::ColorRange => OpKind::Color,
        MaskKind::BrightnessRange => OpKind::Brightness,
    }
}

/// The footprint of a node kind with `reach_px` of its own (a Film node's halo), or `None` for the Input and the Output.
#[must_use]
pub fn node_footprint(kind: NodeKind, reach_px: usize) -> Option<Declared> {
    step_kind(kind).map(|k| k.declared(reach_px))
}

/// The footprint of a mask kind at a render whose raster planes reach `raster_reach_px`.
#[must_use]
pub fn mask_footprint(kind: MaskKind, raster_reach_px: usize) -> Declared {
    op_kind(kind).declared(raster_reach_px)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ym_color::schedule::Footprint;

    #[test]
    fn every_kind_has_a_footprint() {
        for k in NodeKind::ALL {
            let d = node_footprint(k, 7);
            assert_eq!(
                d.is_some(),
                !matches!(k, NodeKind::Input | NodeKind::Output)
            );
        }
        for k in MaskKind::ALL {
            let spatial = matches!(mask_footprint(k, 5).footprint, Footprint::Spatial { .. });
            assert_eq!(
                spatial,
                matches!(
                    k,
                    MaskKind::Object
                        | MaskKind::Subject
                        | MaskKind::People
                        | MaskKind::Background
                        | MaskKind::Brush
                ),
                "{k:?}"
            );
        }
    }
}
