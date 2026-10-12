//! Everything per pixel after C1 to C6: the CPU reference of the colour chain from the fused
//! matrix to display code values (stages C7 to C13) (`docs/02-colour-pipeline.md` sections Stage
//! order, C7 to C13).
//!
//! This is the function the CPU fallback path runs and every GPU kernel is parity-tested against.
//! It reuses the finished stage modules and adds no colour maths of its own:
//!
//! | Stage | Module |
//! |---|---|
//! | C1 to C6 (fused 3x3, supplied by the caller) | [`crate::chain`] (the matrix is taken as `[[f64; 3]; 3]`) |
//! | C7 highlight roll-off and low-light toe | [`crate::tone`] |
//! | C8 LogC4 encode | [`crate::logc`] |
//! | C9 primary grade | [`crate::grade`] |
//! | C10 look (CDL, LUT, intensity) | [`crate::look`] |
//! | C11 curve | [`crate::curve`] |
//! | C12 display transform | [`DisplayTransform`]: [`crate::aces2`] or an imported ARRI display LUT |
//! | C13 display encode and dither | [`crate::encode`] |
//!
//! # Pieces
//!
//! * [`DisplayTransform`] (C12): [`DisplayTransform::Aces2`], or [`DisplayTransform::ArriDisplayLut`]
//!   with the LUT's input encoding ([`DisplayLutInput`]) and output ([`DisplayLutOutput`]: primaries
//!   and transfer, parsed from the LUT's documented target such as `rec709-g24`).
//! * [`OutputTarget`]: Viewport (Display P3), sRGB, Display P3 or Rec.2020, the rows of the C12 table.
//! * [`ChainParams`] and [`ColorChain`]: an immutable, prepared per-photo chain. Methods work on one
//!   pixel ([`ColorChain::apply_pixel`], [`ColorChain::apply_to_display_linear`],
//!   [`ColorChain::apply_dng_pixel`], [`ColorChain::apply_before_pixel`], [`ColorChain::trace`]) or on
//!   whole buffers ([`Frame`], [`Layout`]) split into row bands over scoped threads.
//!
//! # Order of the stages
//!
//! ```text
//! camera RGB --M--> linear AWG4 --C7--> --C8--> LogC4 --C9--> --C10 look--> --C11 curve--> --C12--> --C13--> code values
//!                   (M is C1..C6)       L = luma of the linear values entering C8 (docs/02, C9)
//! ```
//!
//! A look that **includes display rendering** (LogC3 to Rec.709) replaces C12 and changes the order
//! (`docs/02`, C10): `C9 -> C11 -> { our C12, look LUT } -> blend in display-linear`. The curve runs
//! before the look, both branches start from the same C11 output, and the blend uses the look
//! intensity. [`StageTrace`] shows where each value lands for both orders.
//!
//! # Conventions
//!
//! * **American spelling in identifiers.** The chain type is [`ColorChain`], not `ColourChain`.
//! * **f64 throughout.** Buffers are f32 on the way in and out (the layout of
//!   `ym_gpu::cpu::RgbImage`: planar, all red samples, then green, then blue) and every pixel is
//!   computed in f64.
//! * **Deterministic.** Pixels are independent, so the result does not depend on the thread count,
//!   the band size or the order bands run in. Dither uses absolute output pixel coordinates (the
//!   `origin` argument of the 8-bit methods), so tiled and whole-image renders agree bit for bit.
//! * **Non-finite input.** A NaN camera sample counts as 0 and infinities are limited to +-1e30, so
//!   the output is always a defined code value.
//! * **Before view.** See [`BeforePoint`] and [`ColorChain::apply_before_pixel`].

mod chain;
mod frame;
mod transform;

pub use chain::{BeforePoint, ChainParams, ColorChain, LookSetting, StageTrace, View};
pub use frame::{Frame, Layout};
pub use transform::{
    ArriDisplayLut, DisplayLutInput, DisplayLutOutput, DisplayLutPrimaries, DisplayLutTransfer,
    DisplayTransform,
};

use crate::aces2::Aces2Output;
use crate::encode::OutputSpace;

/// Why a chain, a display LUT description or a buffer was rejected.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum DisplayError {
    /// The fused colour matrix has a NaN or infinite entry.
    #[error("the fused colour matrix contains a value that is not a finite number")]
    NonFiniteMatrix,
    /// The look includes its own display rendering, which a DNG cannot hold (`docs/09`).
    #[error(
        "this look includes its own display rendering, so it can't go into a DNG; pick another look or export TIFF"
    )]
    DngUnavailable,
    /// The display LUT's target is not one the viewport and export paths can render.
    #[error("display LUT target {target:?} is not supported: {reason}")]
    UnsupportedTarget {
        /// The target text that was parsed.
        target: String,
        /// What is not supported.
        reason: &'static str,
    },
    /// A display LUT's gamma is outside 1 to 4 or not a number.
    #[error("display LUT gamma {0} is outside 1 to 4")]
    BadGamma(f64),
    /// A buffer does not hold three samples per pixel.
    #[error("buffer holds {got} samples but {width} x {height} pixels need {expected}")]
    BadBuffer {
        /// Samples supplied.
        got: usize,
        /// Samples required.
        expected: usize,
        /// Image width in pixels.
        width: usize,
        /// Image height in pixels.
        height: usize,
    },
    /// The image is so large that its sample count overflows `usize`.
    #[error("an image of {width} x {height} pixels is too large")]
    TooLarge {
        /// Image width in pixels.
        width: usize,
        /// Image height in pixels.
        height: usize,
    },
}

/// One of the four display outputs of the table in `docs/02` section C12.
///
/// `Viewport` and `DisplayP3` are the same colours (P3-D65 limited, Display P3 primaries, sRGB
/// transfer); they differ only in what the caller means by them (the on-screen viewport against a
/// P3 export file).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OutputTarget {
    /// The viewport: Display P3 primaries, sRGB transfer.
    Viewport,
    /// sRGB export: Rec.709 primaries, sRGB transfer.
    Srgb,
    /// Display P3 export: P3-D65 primaries, sRGB transfer.
    DisplayP3,
    /// Rec.2020 export: Rec.2020 primaries, BT.1886 (gamma 2.4) transfer.
    Rec2020,
}

impl OutputTarget {
    /// All four targets, in the order of the spec table.
    pub const ALL: [OutputTarget; 4] = [
        OutputTarget::Viewport,
        OutputTarget::Srgb,
        OutputTarget::DisplayP3,
        OutputTarget::Rec2020,
    ];

    /// The ACES 2 output with the same limiting primaries, encoding primaries and transfer.
    #[must_use]
    pub fn aces2(self) -> Aces2Output {
        match self {
            Self::Viewport => Aces2Output::Viewport,
            Self::Srgb => Aces2Output::Srgb,
            Self::DisplayP3 => Aces2Output::DisplayP3,
            Self::Rec2020 => Aces2Output::Rec2020,
        }
    }

    /// The C13 colour space (primaries and transfer) this target is encoded in.
    #[must_use]
    pub fn space(self) -> OutputSpace {
        match self {
            Self::Viewport | Self::DisplayP3 => OutputSpace::DisplayP3,
            Self::Srgb => OutputSpace::Srgb,
            Self::Rec2020 => OutputSpace::Rec2020,
        }
    }

    /// A stable lower-case name (`viewport`, `srgb`, `display_p3`, `rec2020`).
    #[must_use]
    pub fn name(self) -> &'static str {
        self.aces2().name()
    }
}

impl From<Aces2Output> for OutputTarget {
    fn from(output: Aces2Output) -> Self {
        match output {
            Aces2Output::Viewport => Self::Viewport,
            Aces2Output::Srgb => Self::Srgb,
            Aces2Output::DisplayP3 => Self::DisplayP3,
            Aces2Output::Rec2020 => Self::Rec2020,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn targets_map_to_the_matching_aces2_output_and_space() {
        for t in OutputTarget::ALL {
            assert_eq!(OutputTarget::from(t.aces2()), t);
        }
        assert_eq!(OutputTarget::Viewport.space(), OutputSpace::DisplayP3);
        assert_eq!(OutputTarget::DisplayP3.space(), OutputSpace::DisplayP3);
        assert_eq!(OutputTarget::Srgb.space(), OutputSpace::Srgb);
        assert_eq!(OutputTarget::Rec2020.space(), OutputSpace::Rec2020);
        assert_eq!(OutputTarget::Rec2020.name(), "rec2020");
    }
}
