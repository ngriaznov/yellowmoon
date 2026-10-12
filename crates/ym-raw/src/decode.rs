//! Dispatch on the TIFF compression value to the decoders in the submodules (`docs/04-raw-decoding.md` section Decoders by TIFF compression).
//!
//! [`decode`] is the entry point (`ym_raw::decode::decode`). It probes the file
//! ([`crate::metadata::probe`]), looks at [`DecodeKind`] and hands the work to the matching decoder
//! module:
//!
//! | [`DecodeKind`] | Module | Result |
//! |---|---|---|
//! | `Uncompressed14`, `Uncompressed12` | [`uncompressed`] | 16-bit words or packed 12-bit samples, as stored |
//! | `Craw` | [`craw`] | camera curve applied, undithered, plus the `qstep` map |
//! | `Arw1` | [`arw1`] | the Huffman-coded 12-bit samples of the A200 to A390 |
//! | `LosslessL` | [`ljpeg`] | Bayer mosaic from tiled four-component lossless JPEG |
//! | `LosslessMS` | [`ycbcr`] | camera RGB (white balance undone), [`Layout::Ycbcr`] |
//! | `Arw6`, `Unknown` | [`arw6`] | `RawError::Unsupported` |
//!
//! A decoder either returns the whole image or an error: a truncated or malformed file is
//! [`RawError::Corrupt`] and never yields partially decoded pixels.
//!
//! # Scale and layout of the result
//!
//! `data`, `black`, `white` and `qstep` share one scale: the samples as the file stores them after
//! the camera's own curve (14-bit for the 14-bit modes, lossless L and cRAW; 12-bit for the 12-bit
//! modes). `data` covers the whole readout (`meta.width` x `meta.height`, margins included);
//! `Layout::Bayer::active_area` and `crop` say what part is image. For [`Layout::Ycbcr`] (Lossless M
//! and S) `data` holds interleaved camera RGB: three samples per pixel, `data.width` is
//! `3 * pixel width` ([`RawImage::pixel_width`]), and `black` and `white` are ordered red, green,
//! green, blue instead of CFA raster order. Such an image is not a mosaic and skips demosaicing.

use std::path::Path;

use half::f16;

use crate::error::RawError;
use crate::layout::{DecodeKind, Layout, Plane};
use crate::metadata::{self, JpegRef, LensTags, RawMetadata};
use crate::tiff::{ByteSource, FileSource, MemSource};

pub mod arw1;
pub mod arw6;
pub mod craw;
pub mod ljpeg;
pub mod uncompressed;
pub mod ycbcr;

pub use ycbcr::{ChromaUpsampling, YcbcrOptions};

/// A decoded raw image (`docs/04-raw-decoding.md` section Output).
///
/// Decoders never dither: `data` holds exactly the sensor values the camera wrote, linear after
/// the camera's own curve. `black`, `white` and `qstep` are on the same scale as `data`.
#[derive(Clone, Debug)]
pub struct RawImage {
    /// Camera, exposure, lens, timestamps, orientation and the container details.
    pub meta: RawMetadata,
    /// How the samples are arranged (Bayer mosaic or camera-processed YCbCr).
    pub layout: Layout,
    /// Undithered sensor values, linear after the camera curve. For [`Layout::Ycbcr`], interleaved
    /// camera RGB with `3 * pixel width` samples per row.
    pub data: Plane<u16>,
    /// Black level per CFA position (red, green, green, blue for [`Layout::Ycbcr`]).
    pub black: [f32; 4],
    /// White level per CFA position (red, green, green, blue for [`Layout::Ycbcr`]).
    pub white: [f32; 4],
    /// As-shot white balance gains per CFA position (SR2 0x7313 or 0x7303), as stored (green about
    /// 1024).
    pub wb_as_shot: Option<[f32; 4]>,
    /// Quantisation step per pixel in output DN (cRAW and ARW 6.0 only), same size as `data`.
    pub qstep: Option<Plane<f16>>,
    /// Lens-correction tables (tags 0x7032, 0x7035, 0x7037).
    pub lens: Option<LensTags>,
    /// cRAW tone-curve knots as stored in tag 0x7010.
    pub tone_curve: Option<[u16; 4]>,
    /// Location of the embedded full-size JPEG preview.
    pub preview: Option<JpegRef>,
}

impl RawImage {
    /// True for camera-processed files (Lossless M and S): `data` is RGB, not a mosaic, and the
    /// pipeline skips demosaicing and runs the REDUCED path (`docs/04` section Lossless M and S).
    pub fn is_reduced(&self) -> bool {
        matches!(self.layout, Layout::Ycbcr { .. })
    }

    /// Samples per pixel in `data`: 1 for a mosaic, 3 for [`Layout::Ycbcr`].
    pub fn channels(&self) -> usize {
        if self.is_reduced() { 3 } else { 1 }
    }

    /// Width of the image in pixels (`data.width / channels()`).
    pub fn pixel_width(&self) -> usize {
        self.data.width / self.channels()
    }

    /// Height of the image in pixels.
    pub fn pixel_height(&self) -> usize {
        self.data.height
    }
}

/// Switches of [`decode_with`]. The defaults give the result the pipeline wants.
#[derive(Clone, Copy, Debug, Default)]
pub struct DecodeOptions {
    /// Lossless M and S: chroma upsampling and white-balance undo.
    pub ycbcr: YcbcrOptions,
}

/// Decodes the raw image of the file at `path`.
///
/// Errors from the probe (`NotArw`, `Arq`, `Corrupt`, `Io`) are returned as they are; ARW 6.0 and
/// layouts this crate cannot classify are [`RawError::Unsupported`].
pub fn decode(path: impl AsRef<Path>) -> Result<RawImage, RawError> {
    decode_with(path, &DecodeOptions::default())
}

/// Like [`decode`], with explicit options.
pub fn decode_with(path: impl AsRef<Path>, options: &DecodeOptions) -> Result<RawImage, RawError> {
    let source = FileSource::open(path.as_ref())?;
    decode_source(&source, options)
}

/// Decodes an ARW held in memory.
pub fn decode_bytes(bytes: &[u8], options: &DecodeOptions) -> Result<RawImage, RawError> {
    decode_source(&MemSource(bytes), options)
}

/// Decodes an ARW from any [`ByteSource`].
pub fn decode_source(
    source: &dyn ByteSource,
    options: &DecodeOptions,
) -> Result<RawImage, RawError> {
    let meta = metadata::probe_source(source)?;
    decode_with_metadata(source, meta, options)
}

/// Dispatches on `meta.decode`.
pub fn decode_with_metadata(
    source: &dyn ByteSource,
    meta: RawMetadata,
    options: &DecodeOptions,
) -> Result<RawImage, RawError> {
    let mut black = meta.black;
    let mut white = meta.white;
    let mut qstep = None;
    let layout;
    let data = match meta.decode {
        DecodeKind::Uncompressed14 => {
            layout = meta.layout();
            uncompressed::decode(source, &meta, uncompressed::Packing::Words16)?
        }
        DecodeKind::Uncompressed12 => {
            layout = meta.layout();
            let packing = if meta.compression == metadata::COMPRESSION_NONE {
                uncompressed::Packing::Words16
            } else {
                uncompressed::Packing::Packed12
            };
            uncompressed::decode(source, &meta, packing)?
        }
        DecodeKind::Craw => {
            layout = meta.layout();
            let image = craw::decode(source, &meta)?;
            qstep = Some(image.qstep);
            image.data
        }
        DecodeKind::Arw1 => {
            layout = meta.layout();
            arw1::decode(source, &meta)?
        }
        DecodeKind::LosslessL => {
            layout = meta.layout();
            ljpeg::decode_lossless_l(source, &meta)?
        }
        DecodeKind::LosslessMS { .. } => {
            let image = ycbcr::decode(source, &meta, &options.ycbcr)?;
            layout = Layout::Ycbcr {
                subsampling: image.subsampling,
            };
            black = image.black;
            white = image.white;
            image.rgb
        }
        DecodeKind::Arw6 => {
            #[cfg(feature = "arw6")]
            {
                layout = meta.layout();
                arw6::decode(source, &meta)?
            }
            #[cfg(not(feature = "arw6"))]
            {
                return Err(arw6::unsupported(&meta));
            }
        }
        DecodeKind::Unknown { .. } => {
            return Err(RawError::Unsupported {
                camera: if meta.model.is_empty() {
                    meta.make.clone()
                } else {
                    meta.model.clone()
                },
                variant: meta.decode,
            });
        }
    };
    Ok(RawImage {
        layout,
        data,
        black,
        white,
        wb_as_shot: meta.wb_as_shot,
        qstep,
        lens: meta.lens_corrections.clone(),
        tone_curve: meta.tone_curve,
        preview: meta.preview,
        meta,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testgen::{ArwBuilder, ContainerOnly};

    fn write(dir: &tempfile::TempDir, name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let path = dir.path().join(name);
        std::fs::write(&path, bytes).expect("write test file");
        path
    }

    #[test]
    fn arw6_files_get_the_spec_message() {
        let dir = tempfile::tempdir().expect("tempdir");
        let file = ArwBuilder::new("ILCE-7M5", 16, 16)
            .container_only(ContainerOnly::arw6(12))
            .build(&[]);
        let path = write(&dir, "a7v.ARW", &file);
        let error = decode(&path).expect_err("unsupported");
        assert_eq!(
            error.to_string(),
            "ARW 6.0 compressed files from the A7 V and A7R VI aren't supported yet. \
             Lossless files from these cameras work. We're working on it."
        );
        assert!(matches!(error, RawError::Unsupported { .. }));
    }

    #[test]
    fn probe_errors_pass_through_decode() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = write(&dir, "x.jpg", b"\xff\xd8\xff\xe0\0\x10JFIF\0not a raw");
        assert!(matches!(decode(&path), Err(RawError::NotArw)));
        assert!(matches!(
            decode(dir.path().join("missing.ARW")),
            Err(RawError::Io(_))
        ));
    }
}
