//! Auto crop (`docs/20-auto-crop.md`): the straightener, the subject map and the frame proposer.
//!
//! [`analyse`] looks at one picture of a photo (Apple Auto's 768 px picture, `docs/19` §3) with the
//! camera's hints and keeps what every later request needs: the straighten answer and the subject
//! and distraction maps. [`Analysis::propose`] then makes the frames for one aspect, quarter turn
//! and angle in a few milliseconds.
//!
//! Everything is f64 on one thread with fixed orders and explicit tie rules, so the same picture
//! and hints give the same answer on every platform (hard rule 8).

#![warn(missing_docs)]

mod frame;
mod lines;
mod picture;
mod saliency;
mod straighten;
mod subject;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod testutil;

pub use frame::{Frame, FrameRequest, Proposal, Ratio, Rect, largest_frame};
pub use straighten::Straighten;

use subject::SubjectMaps;

/// Why a picture was refused.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ComposeError {
    /// The byte count is not `width × height × 3`, or a side is 0.
    #[error("the picture is {width} x {height} pixels but holds {len} bytes")]
    Size {
        /// Width in pixels.
        width: usize,
        /// Height in pixels.
        height: usize,
        /// Bytes given.
        len: usize,
    },
}

/// One picture of a photo: 8-bit sRGB, 3 bytes per pixel, rows top to bottom (`docs/20` §4).
#[derive(Debug, Clone, Copy)]
pub struct Picture<'a> {
    width: usize,
    height: usize,
    rgb: &'a [u8],
}

impl<'a> Picture<'a> {
    /// A picture of `width × height` pixels.
    ///
    /// # Errors
    ///
    /// [`ComposeError::Size`] when a side is 0 or `rgb` is not `width × height × 3` bytes.
    pub fn new(width: usize, height: usize, rgb: &'a [u8]) -> Result<Self, ComposeError> {
        if width == 0
            || height == 0
            || width.checked_mul(height).and_then(|n| n.checked_mul(3)) != Some(rgb.len())
        {
            return Err(ComposeError::Size {
                width,
                height,
                len: rgb.len(),
            });
        }
        Ok(Self { width, height, rgb })
    }

    /// Width in pixels.
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Height in pixels.
    #[must_use]
    pub fn height(&self) -> usize {
        self.height
    }

    /// The bytes, `width × height × 3`.
    #[must_use]
    pub fn rgb(&self) -> &'a [u8] {
        self.rgb
    }
}

/// What the camera says about the subject, in the picture's space (`docs/20` §4).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct CameraHint {
    /// The focus point as fractions of the picture's width and height.
    pub point: Option<[f64; 2]>,
    /// The AF box around the point as fractions of the picture's width and height.
    pub box_size: Option<[f64; 2]>,
    /// The photographer placed the AF area away from the centre (Flexible Spot, Zone, ...).
    pub placed: bool,
}

/// What [`analyse`] found in a picture; [`Analysis::propose`] makes frames from it.
#[derive(Debug, Clone)]
pub struct Analysis {
    width: usize,
    height: usize,
    straighten: Straighten,
    /// Where the horizon crosses the picture's centre column, as a fraction of its height.
    horizon: Option<f64>,
    maps: SubjectMaps,
    hint: CameraHint,
}

/// Looks at a picture with the camera's hints: the straightener (`docs/20` §5) and the subject map
/// (§6).
#[must_use]
pub fn analyse(picture: &Picture<'_>, hint: &CameraHint) -> Analysis {
    let (straighten, horizon) = straighten::straighten(picture);
    let maps = subject::subject_maps(picture, hint);
    Analysis {
        width: picture.width,
        height: picture.height,
        straighten,
        horizon,
        maps,
        hint: *hint,
    }
}

impl Analysis {
    /// The straightener's answer.
    #[must_use]
    pub fn straighten(&self) -> Straighten {
        self.straighten
    }

    /// The straighten angle Auto uses: the answer's angle when it is [`Straighten::Rotate`], else 0.
    #[must_use]
    pub fn angle_deg(&self) -> f64 {
        match self.straighten {
            Straighten::Rotate { angle_deg } => angle_deg,
            Straighten::Level | Straighten::Unsure => 0.0,
        }
    }

    /// Whether the subject came from the camera's focus point (else from the picture alone).
    #[must_use]
    pub fn subject_from_camera(&self) -> bool {
        self.hint.point.is_some()
    }

    /// The picture's size in pixels.
    #[must_use]
    pub fn picture_size(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    /// The frames for one request (`docs/20` §7): best first, at most 3, never empty.
    #[must_use]
    pub fn propose(&self, request: &FrameRequest) -> Proposal {
        frame::propose(
            self.width,
            self.height,
            &self.maps,
            &self.hint,
            self.horizon,
            request,
        )
    }
}
