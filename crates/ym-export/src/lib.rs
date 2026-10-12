//! Export: the JPEG, TIFF, PNG and DNG writers, ICC profiles, EXIF building, XMP and file naming.
//!
//! App and CLI share this code path, so the same inputs and settings give byte-identical files.
//! Depends on `ym-color` for the display encode and colour spaces; it does not depend on `ym-raw` or
//! `ym-core`, so its inputs are plain structs that `ym-core` fills in.
//!
//! Start at [`request::write_export`] (one photo to one file, atomically) and
//! [`request::estimate_bytes`]. The options live in [`options`], the metadata in [`exif`] and [`xmp`].
//!
//! Specs: `docs/09-export.md` (options, queue, pixel pipeline, colour management, metadata,
//! DNG with the look baked in).
//!
//! This product includes DNG technology under license by Adobe.
//!
//! This software is based in part on the work of the Independent JPEG Group: the forward DCT of the
//! `jpeg-encoder` crate is ported from their software (the crate's licence is
//! `(MIT OR Apache-2.0) AND IJG`). The IJG licence asks for this sentence in the documentation that
//! accompanies a binary, so it also belongs in `NOTICE` and the About window.

#![warn(missing_docs)]

pub mod dng;
pub mod error;
pub mod exif;
pub mod icc;
pub mod jpeg;
pub mod naming;
pub mod options;
pub mod pixels;
pub mod png_writer;
pub mod request;
pub mod resize;
pub mod tiff_writer;
pub mod xmp;

pub use error::ExportError;
pub use options::{
    BitDepth, ColorSpace, ExportFormat, ExportOptions, FormatKind, IfExists, JpegQuality, Size,
    TiffCompression,
};
pub use pixels::{ExportHooks, ImageRef, PixelEncoding};
pub use request::{
    DngInputs, ExportDestination, ExportOutcome, ExportRequest, Metadata, encode_to_vec,
    encode_to_writer, estimate_bytes, write_export,
};
