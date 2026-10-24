//! Lens character: lens data, plans, planes and the CPU references (`docs/21-lens-character.md`).
//!
//! A photo gets the look of another lens: how its edges soften and darken (L2), the anamorphic bend, the
//! lens's distortion and the zoom (L4), and, in later passes, bokeh reshaping (L1) and highlight optics (L3).
//! The crate works on planar `f32` images with their place in the frame ([`ImageRef`]) and depends on
//! `ym-color` only for the camera's spectral curves ([`CameraResponse`]); the GPU kernels live in `ym-gpu`,
//! which will depend on this crate for their parameters.
//!
//! The flow:
//!
//! 1. [`LensLibrary::load`] reads and checks `data/lenses/` once.
//! 2. [`LensPlan::new`] binds a lens, three [`LensAmounts`], a [`FrameInfo`] and a [`CameraResponse`].
//! 3. [`LensPlanes::build`] makes the amount-independent per-photo data once.
//! 4. [`apply`] runs L1 to L3 on any rectangle of the frame, given the input grown by [`halo_px`]; any tiling
//!    gives the same bits as the whole frame.
//! 5. [`LensPlan::warp`] gives the L4 map between the canvas and the photo.
//!
//! With all three amounts at 0 ([`LensAmounts::is_identity`]) nothing runs and [`apply`] returns its input.

#![warn(missing_docs)]

mod amounts;
mod apply;
pub mod consts;
mod data;
mod error;
mod field;
mod image;
mod l1;
mod l3;
mod plan;
mod planes;
mod simd;
mod spectral;
#[cfg(any(test, feature = "test-support"))]
pub mod test_support;
mod warp;

pub use amounts::{LensAmounts, LensKind};
pub use apply::{apply, halo_px};
pub use data::{
    Aperture, AttachmentData, AttachmentField, AttachmentProvenance, Bend, Design, Distortion,
    FieldData, FlareData, GhostData, Glow, LensData, LensInfo, LensLibrary, LensProvenance,
    Starburst, Streak, Veil,
};
pub use error::LensError;
pub use image::{ImageRef, PixelRect};
pub use l1::{Ball, BokehPlanes, debug_samples};
pub use plan::{Axis, FrameInfo, LensPlan};
pub use planes::{LensPlanes, ReshapeReason, ReshapeReport};
pub use spectral::CameraResponse;
pub use warp::LensWarp;
