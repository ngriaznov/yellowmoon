//! The picture as a JPEG (`docs/22` section 6.1): a baseline JPEG, 4:4:4, quality 92, Huffman tables
//! not optimised, no metadata. `jpeg-encoder` then writes one interleaved scan, which every decoder
//! reads, and the same picture gives the same bytes.

use jpeg_encoder::{ColorType, Encoder, SamplingFactor};
use ym_core::apple_auto::AdvisorPicture;

/// The JPEG quality (`docs/22` D13).
pub const QUALITY: u8 = 92;

/// Why a picture cannot be encoded.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum JpegError {
    /// A side is 0.
    #[error("the picture is empty ({width} x {height})")]
    Empty {
        /// Its width in pixels.
        width: usize,
        /// Its height in pixels.
        height: usize,
    },
    /// The byte count is not `width * height * 3`.
    #[error("a {width} x {height} picture needs {expected} bytes and has {actual}")]
    Size {
        /// Its width in pixels.
        width: usize,
        /// Its height in pixels.
        height: usize,
        /// `width * height * 3`.
        expected: usize,
        /// What the picture holds.
        actual: usize,
    },
    /// A side does not fit a JPEG's 16 bits.
    #[error("a {width} x {height} picture is too large for a JPEG")]
    TooLarge {
        /// Its width in pixels.
        width: usize,
        /// Its height in pixels.
        height: usize,
    },
    /// The encoder failed.
    #[error("the JPEG encoder failed: {0}")]
    Encoder(String),
}

/// Encodes `picture` (8-bit sRGB, 3 bytes per pixel, rows top to bottom) as `docs/22` section 6.1
/// describes. A picture whose byte count is not `width * height * 3` is refused before encoding.
pub fn encode(picture: &AdvisorPicture) -> Result<Vec<u8>, JpegError> {
    let (width, height) = (picture.width, picture.height);
    if width == 0 || height == 0 {
        return Err(JpegError::Empty { width, height });
    }
    let expected = width
        .checked_mul(height)
        .and_then(|n| n.checked_mul(3))
        .ok_or(JpegError::TooLarge { width, height })?;
    if picture.rgb.len() != expected {
        return Err(JpegError::Size {
            width,
            height,
            expected,
            actual: picture.rgb.len(),
        });
    }
    let (Ok(w), Ok(h)) = (u16::try_from(width), u16::try_from(height)) else {
        return Err(JpegError::TooLarge { width, height });
    };
    let mut out = Vec::with_capacity(expected / 8 + 1024);
    let mut encoder = Encoder::new(&mut out, QUALITY);
    encoder.set_progressive(false);
    encoder.set_optimized_huffman_tables(false);
    encoder.set_sampling_factor(SamplingFactor::R_4_4_4);
    encoder
        .encode(&picture.rgb, w, h, ColorType::Rgb)
        .map_err(|e| JpegError::Encoder(e.to_string()))?;
    Ok(out)
}
