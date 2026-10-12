//! Sony ARW decoding: our own Apache-2.0 decoder.
//!
//! Walks the TIFF container, reads EXIF, the Sony maker note and the enciphered SR2 private data,
//! and decodes every raw variant (uncompressed, cRAW, lossless L, the reduced M/S path and, behind
//! a feature, ARW 6.0) into CPU buffers. It has no GPU dependency. rawler and LibRaw are test
//! oracles only (`tools/oracle`) and must never be linked here.
//!
//! Specs: `docs/04-raw-decoding.md` (container, SR2, decoders, errors, tests) and
//! `docs/03-camera-profiles.md` section Detection (the `CameraMatch` result).

#![warn(missing_docs)]

pub mod decode;
pub mod error;
pub mod exif;
pub mod layout;
pub mod makernote;
pub mod metadata;
pub mod quirks;
pub mod sr2;
#[cfg(any(test, feature = "testgen"))]
pub mod testgen;
pub mod tiff;
