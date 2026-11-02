//! What the renderer tells its listener about the Subject and People worker (`docs/23-nodes.md` section 4.5). The work
//! itself is in `graph/vision.rs`; the provider that makes the masks, and the probe, are `crate::vision`.

use crate::settings::MaskKind;

/// What the background worker is doing for a photo, as the renderer's listener hears it
/// ([`Renderer::set_vision_listener`](super::Renderer::set_vision_listener)).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VisionPhase {
    /// A worker started for the photo: the probe, the picture and the requests are on their way. The rows say
    /// "Finding…".
    Started,
    /// The worker ended its work for the photo. A frame that selected nothing for want of a mask should be drawn again,
    /// and the rows whose request found nothing say "Nothing found".
    Finished {
        /// The requests whose newest result for the photo found nothing (as mask kinds).
        nothing_found: Vec<MaskKind>,
    },
}
