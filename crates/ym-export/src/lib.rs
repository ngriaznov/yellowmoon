//! Export: the JPEG, TIFF, PNG and DNG writers, ICC profiles, EXIF copying, XMP and file naming.
//!
//! App and CLI share this code path, so the same inputs and settings give byte-identical files.
//! Depends on `ym-color` for the display encode and colour spaces.
//!
//! Specs: `docs/09-export.md` (options, queue, pixel pipeline, colour management, metadata,
//! DNG with the look baked in).

#![warn(missing_docs)]

pub mod dng;
pub mod exif;
pub mod icc;
pub mod jpeg;
pub mod naming;
pub mod options;
pub mod png_writer;
pub mod resize;
pub mod tiff_writer;
pub mod xmp;
