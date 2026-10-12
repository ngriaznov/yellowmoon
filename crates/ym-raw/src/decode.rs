//! Dispatch on the TIFF compression value to the decoders in the submodules (`docs/04-raw-decoding.md` section Decoders by TIFF compression).

pub mod arw6;
pub mod craw;
pub mod ljpeg;
pub mod uncompressed;
pub mod ycbcr;
