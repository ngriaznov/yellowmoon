//! ARW 6.0 Compressed RAW 2 detection, and the place for the decoder behind the `arw6` feature (`docs/04-raw-decoding.md` section ARW 6.0).
//!
//! There is no ARW 6.0 decoder yet. The codec (a tiled, packetised wavelet coder built on Sony's
//! LLVC, SMPTE RDD 34) stays gated on the \[HUMAN\] task M2-H3, the question to Sony about IP for
//! RDD 34-based decoding, and a permissively licensed implementation (lightcraft) is to be studied
//! when the gate clears. Until then every file with TIFF compression 32766 is detected by
//! [`crate::metadata::classify`] as [`DecodeKind::Arw6`] and refused with the message the spec
//! prescribes (`RawError::Unsupported`, "ARW 6.0 compressed files from the A7 V and A7R VI aren't
//! supported yet. Lossless files from these cameras work. We're working on it.").
//!
//! The `arw6` cargo feature is reserved for the decoder and is off in every build today. With it
//! on, `decode` is the hook [`super::decode_with_metadata`] calls for ARW 6.0 files; it still
//! returns the same `Unsupported` error, so turning the feature on changes nothing until a decoder
//! is written here.

use crate::error::RawError;
use crate::layout::DecodeKind;
use crate::metadata::RawMetadata;

/// True when the file's raw data is ARW 6.0 Compressed RAW 2 (TIFF compression 32766).
pub fn is_arw6(meta: &RawMetadata) -> bool {
    meta.decode == DecodeKind::Arw6
}

/// The error for an ARW 6.0 file: [`RawError::Unsupported`] naming the camera.
pub fn unsupported(meta: &RawMetadata) -> RawError {
    RawError::Unsupported {
        camera: if meta.model.is_empty() {
            meta.make.clone()
        } else {
            meta.model.clone()
        },
        variant: DecodeKind::Arw6,
    }
}

/// The decoder hook behind the `arw6` feature. Not implemented: always [`unsupported`].
#[cfg(feature = "arw6")]
pub fn decode(
    _src: &dyn crate::tiff::ByteSource,
    meta: &RawMetadata,
) -> Result<crate::layout::Plane<u16>, RawError> {
    Err(unsupported(meta))
}
