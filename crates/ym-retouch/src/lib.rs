//! Face skin retouching (`docs/27-skin-retouch.md`): blemish removal, dark circles, even skin, skin
//! grading and teeth whitening.
//!
//! This crate holds the types and rules of sections 2 to 5 of the spec: the image, mask and
//! settings types, the model traits and their deterministic mocks, and the mask arithmetic, the
//! feather and the mask builders. Colour maths is `f64` on this CPU reference path; masks are
//! `u8` with integer arithmetic. Image-sized passes run over rows with `rayon` and write disjoint
//! rows, so results never depend on the thread count (hard rule 8).

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod blemish;
pub mod classical;
mod error;
mod face_parts;
mod face_raster;
mod face_tables;
mod face_topology;
mod feather;
mod guided;
mod lowfreq;
mod mask;
#[cfg(feature = "mediapipe")]
pub mod mediapipe;
#[cfg(any(test, feature = "mock"))]
pub mod mock;
mod models;
mod pipeline;
mod types;

pub use blemish::{
    BlemishMap, BlemishMapCache, BlemishMapKey, CHROMA_GUARD_MIN_SIDE_PX, GAIN_MAX, GAIN_MIN,
    TARGET_FACE_PX, apply_blemish, apply_gain, blemish_gain, blemish_map, chroma_confidence,
    chroma_pull, encode_channel, gain_strength, threshold_strength,
};
pub use classical::{ClassicalBeautyMarks, ClassicalProposer};
pub use error::{Result, RetouchError};
pub use feather::feather;
pub use lowfreq::{
    LowFreqParams, ToolMasks, blend_pixel, blend_weight, contrast_target, dark_circle_k_sp,
    dark_circles_target, even_skin_target, mask_to_working, masked_blur_combine,
    retouch_low_frequency, retouch_tones, shadow_desaturation_pixel, shine_removal_texture,
    teeth_pixel,
};
pub use mask::{
    FaceMasks, blemish_mask, dark_circle_mask, even_skin_mask, face_masks, grading_mask, max,
    max_masks, mul, mul_inv, mul_inv_masks, mul_masks, punch_hole, skin_mask, smoothstep,
    teeth_gate, teeth_mask,
};
pub use models::{
    BeautyMarkDetector, BlemishProposer, FaceMesh, FaceParser, FacePart, MeshTable, Models,
};
pub use pipeline::{RetouchCache, prepare_skin, retouch_skin};
pub use types::{
    BoundedMask, Face, LinearAwg4Image, Mask, Rect, SkinSettings, check_render_scale,
    face_extent_px, face_size_px,
};
