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
    /// `render_scale` is not in `0.0001..=1`.
    #[error("the render scale {0} is outside 0.0001 to 1")]
    RenderScale(f32),
    /// A mask is too large to feather within the memory limit.
    #[error("a mask of {width} x {height} pixels is too large to feather")]
    TooLarge {
        /// Width of the working mask in pixels.
        width: u64,
        /// Height of the working mask in pixels.
        height: u64,
    },
    /// A model failed.
    #[error("a model failed: {0}")]
    Model(String),
}

/// The result type of the crate.
pub type Result<T> = std::result::Result<T, RetouchError>;
