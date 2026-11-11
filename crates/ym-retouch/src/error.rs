//! The error type of the crate.

/// Why a retouch step was refused or failed. Nothing in this crate panics on bad input.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum RetouchError {
    /// A buffer does not hold `width x height` values, or a side is 0.
    #[error("the {what} is {width} x {height} pixels but holds {len} values")]
    Size {
        /// What was checked ("image", "mask").
        what: &'static str,
        /// Width in pixels.
        width: u32,
        /// Height in pixels.
        height: u32,
        /// Values given.
        len: usize,
    },
    /// A bounded mask's region and its mask have different sizes.
    #[error("the mask region is {region:?} pixels but its mask is {mask:?} pixels")]
    RegionMismatch {
        /// Region size (width, height).
        region: (u32, u32),
        /// Mask size (width, height).
        mask: (u32, u32),
    },
    /// Two masks that must match in size do not.
    #[error("the masks differ in size: {left:?} and {right:?} pixels")]
    MaskSizeMismatch {
        /// First mask (width, height).
        left: (u32, u32),
        /// Second mask (width, height).
        right: (u32, u32),
    },
    /// `render_scale` is not in `0.0001..=16`.
    #[error("the render scale {0} is outside 0.0001 to 16")]
    RenderScale(f32),
    /// A mask is too large to feather within the memory limit.
    #[error("a mask of {width} x {height} pixels is too large to feather")]
    TooLarge {
        /// Width of the working mask in pixels.
        width: u64,
        /// Height of the working mask in pixels.
        height: u64,
    },
    /// The working image does not match `full_size_px * render_scale` (within 2 px).
    #[error(
        "the working image is {image:?} pixels but the full size {full:?} at render scale {scale} gives about {expected:?}"
    )]
    WorkingSize {
        /// Working image size (width, height).
        image: (u32, u32),
        /// Full-resolution size (width, height).
        full: (u32, u32),
        /// The render scale.
        scale: f32,
        /// The size the render scale implies.
        expected: (u32, u32),
    },
    /// The masks or blemish maps this run needs are not in the cache, and the image given is not
    /// the full-resolution one they could be built from. Call `prepare_skin` first.
    #[error(
        "the skin masks or blemish maps are not prepared for these faces; prepare them from the full-resolution image first"
    )]
    NotPrepared,
    /// A model failed.
    #[error("a model failed: {0}")]
    Model(String),
    /// A part of the working image cannot be retouched alone (`retouch_skin_region`): `why` says what is wrong with it.
    #[error("this part of the picture can't be retouched alone: {why}")]
    Part {
        /// What is wrong.
        why: &'static str,
    },
}

/// The result type of the crate.
pub type Result<T> = std::result::Result<T, RetouchError>;
