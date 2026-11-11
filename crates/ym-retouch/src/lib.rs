//! Face skin retouching (`docs/27-skin-retouch.md`): blemish removal, dark circles and even skin.
//!
//! This crate holds the types and rules of sections 2 to 5 of the spec: the image, mask and
//! settings types, the model traits and their deterministic mocks, and the mask arithmetic, the
//! feather and the mask builders. Colour maths is `f64` on this CPU reference path; masks are
//! `u8` with integer arithmetic. Image-sized passes run over rows with `rayon` and write disjoint
//! rows, so results never depend on the thread count (hard rule 8).

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod blemish;
mod error;
mod feather;
mod lowfreq;
mod mask;
#[cfg(any(test, feature = "mock"))]
pub mod mock;
mod models;
mod types;

pub use blemish::{
    BlemishMap, BlemishMapCache, BlemishMapKey, CHROMA_GUARD_MIN_SIDE_PX, GAIN_MAX, GAIN_MIN,
    TARGET_FACE_PX, apply_blemish, apply_gain, blemish_gain, blemish_map, chroma_confidence,
    chroma_pull, encode_channel, gain_strength, threshold_strength,
};
pub use error::{Result, RetouchError};
pub use feather::feather;
pub use lowfreq::{
    LowFreqParams, blend_pixel, blend_weight, dark_circle_k_sp, dark_circles_target,
    even_skin_target, mask_to_working, masked_blur_combine, retouch_low_frequency,
};
pub use mask::{
    FaceMasks, SkinParts, blemish_mask, dark_circle_mask, even_skin_mask, face_masks, max,
    max_masks, mul, mul_inv, mul_inv_masks, mul_masks, punch_hole, skin_parts,
};
pub use models::{
    BeautyMarkDetector, BlemishProposer, FaceMesh, FaceParser, FacePart, MeshTable, Models,
};
pub use types::{
    BoundedMask, Face, LinearAwg4Image, Mask, Rect, SkinSettings, check_render_scale,
    face_extent_px, face_size_px,
};
