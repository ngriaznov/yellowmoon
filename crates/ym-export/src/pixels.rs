//! Pixel buffers, band-wise display encoding, and the cancel and progress hooks the writers share.
//!
//! Images come in as interleaved RGB `f32` (see [`PixelEncoding`]). The writers never hold a second
//! full-size copy: they convert a band of rows at a time (in parallel, row by row, which keeps the
//! result independent of the thread count) and hand the band to the encoder.

use rayon::prelude::*;
use ym_color::encode::{self, OutputSpace};

use crate::error::ExportError;

/// What the `f32` values of an [`ImageRef`] mean.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PixelEncoding {
    /// Linear light in the primaries of the export colour space, display-referred (the output of the
    /// display transform, stage C12, before the transfer function). `1.0` is white. JPEG, TIFF and PNG.
    DisplayLinear,
    /// Already encoded with the colour space's transfer function (stage C13 without the quantiser),
    /// `0.0` to `1.0`. A resize decodes it to linear light first.
    DisplayEncoded,
    /// Scene-linear ARRI Wide Gamut 4 (the DNG's raw pixels). Not a display image.
    SceneLinearAwg4,
}

/// A borrowed interleaved RGB image of `f32` samples, row-major, no padding.
#[derive(Debug, Clone, Copy)]
pub struct ImageRef<'a> {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// `width * height * 3` samples: R, G, B of the first pixel, then the next, row by row.
    pub rgb: &'a [f32],
    /// What the samples mean.
    pub encoding: PixelEncoding,
}

impl<'a> ImageRef<'a> {
    /// Wraps a buffer, checking its length.
    pub fn new(
        width: u32,
        height: u32,
        rgb: &'a [f32],
        encoding: PixelEncoding,
    ) -> Result<Self, ExportError> {
        let image = Self {
            width,
            height,
            rgb,
            encoding,
        };
        image.validate()?;
        Ok(image)
    }

    /// Checks that the dimensions are non-zero and the buffer has exactly the right length.
    pub fn validate(&self) -> Result<(), ExportError> {
        if self.width == 0 || self.height == 0 {
            return Err(ExportError::invalid("the image has no pixels"));
        }
        let expected = (self.width as usize)
            .checked_mul(self.height as usize)
            .and_then(|n| n.checked_mul(3))
            .ok_or_else(|| ExportError::invalid("the image is too large"))?;
        if self.rgb.len() != expected {
            return Err(ExportError::invalid(format!(
                "a {}x{} image needs {expected} samples, got {}",
                self.width,
                self.height,
                self.rgb.len()
            )));
        }
        Ok(())
    }

    /// Row `y` as `width * 3` samples.
    #[must_use]
    pub fn row(&self, y: usize) -> &'a [f32] {
        let n = self.width as usize * 3;
        &self.rgb[y * n..(y + 1) * n]
    }
}

/// An owned linear-light image (the result of a resize).
#[derive(Debug, Clone, PartialEq)]
pub struct LinearImage {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Interleaved RGB samples.
    pub rgb: Vec<f32>,
}

impl LinearImage {
    /// Borrows the image as display-linear pixels.
    #[must_use]
    pub fn as_ref(&self) -> ImageRef<'_> {
        ImageRef {
            width: self.width,
            height: self.height,
            rgb: &self.rgb,
            encoding: PixelEncoding::DisplayLinear,
        }
    }
}

/// Rows converted per band: about a million pixels, between 8 and 512 rows.
#[must_use]
pub fn band_rows(width: u32) -> usize {
    ((1 << 20) / width.max(1) as usize).clamp(8, 512)
}

/// Encodes rows `y0..y0 + rows` of `image` to 8-bit RGB (stage C13 with blue-noise dither) into `out`.
///
/// The dither uses absolute pixel coordinates, so any band split gives the same bytes. `out` is
/// resized to `rows * width * 3`.
pub fn encode_rows_u8(
    image: &ImageRef<'_>,
    space: OutputSpace,
    y0: usize,
    rows: usize,
    out: &mut Vec<u8>,
) {
    let w = image.width as usize;
    out.clear();
    out.resize(rows * w * 3, 0);
    out.par_chunks_mut(w * 3).enumerate().for_each(|(r, dst)| {
        let y = y0 + r;
        let src = image.row(y);
        match image.encoding {
            PixelEncoding::DisplayEncoded => {
                for (x, (d, s)) in dst.chunks_exact_mut(3).zip(src.chunks_exact(3)).enumerate() {
                    for c in 0..3 {
                        d[c] = encode::dither_8bit(f64::from(s[c]), x, y);
                    }
                }
            }
            _ => {
                for (x, (d, s)) in dst.chunks_exact_mut(3).zip(src.chunks_exact(3)).enumerate() {
                    let lin = [f64::from(s[0]), f64::from(s[1]), f64::from(s[2])];
                    d.copy_from_slice(&encode::encode_8bit_dithered(lin, space, x, y));
                }
            }
        }
    });
}

/// Encodes rows `y0..y0 + rows` of `image` to 16-bit RGB (no dither) into `out`, resized to
/// `rows * width * 3` samples.
pub fn encode_rows_u16(
    image: &ImageRef<'_>,
    space: OutputSpace,
    y0: usize,
    rows: usize,
    out: &mut Vec<u16>,
) {
    let w = image.width as usize;
    out.clear();
    out.resize(rows * w * 3, 0);
    out.par_chunks_mut(w * 3).enumerate().for_each(|(r, dst)| {
        let src = image.row(y0 + r);
        match image.encoding {
            PixelEncoding::DisplayEncoded => {
                for (d, s) in dst.iter_mut().zip(src) {
                    *d = encode::quantize(f64::from(*s), 16);
                }
            }
            _ => {
                for (d, s) in dst.chunks_exact_mut(3).zip(src.chunks_exact(3)) {
                    let lin = [f64::from(s[0]), f64::from(s[1]), f64::from(s[2])];
                    d.copy_from_slice(&encode::encode_quantized(lin, space, 16));
                }
            }
        }
    });
}

/// The finished picture, handed to a writer as encoded rows (stage C13: the transfer function, the
/// dither for 8 bit and the quantiser), in order from the top.
///
/// A writer asks for rows band by band and never twice: `y0` of each call is the row after the last
/// one it received (a source may refuse anything else), and a call for rows past [`height`] is an
/// error. The JPEG writer pulls 8-bit rows, the TIFF and PNG writers 8 or 16 bit depending on the
/// requested depth; a source made for one depth refuses the other.
///
/// [`ImageBands`] encodes an [`ImageRef`] on the CPU (the reference: the bytes of every export before
/// the export tail moved to the device). `ym-core`'s `GpuBands` pulls rows that the device made, so a
/// writer can overlap its own work (compression, disk) with the render of the rows that follow.
///
/// [`height`]: BandSource::height
pub trait BandSource {
    /// Width of the picture in pixels.
    fn width(&self) -> u32;

    /// Height of the picture in pixels.
    fn height(&self) -> u32;

    /// Rows `y0..y0 + rows` as 8-bit interleaved RGB, `rows * width * 3` bytes in `out` (which is
    /// cleared first). The dither, when there is one, uses absolute pixel coordinates.
    ///
    /// # Errors
    ///
    /// [`ExportError::InvalidInput`] for rows outside the picture, a call out of order or a source
    /// that holds 16-bit rows; [`ExportError::Cancelled`] when the export was cancelled while the
    /// rows were being made; any other error of the source (a render that failed).
    fn rows_u8(&mut self, y0: usize, rows: usize, out: &mut Vec<u8>) -> Result<(), ExportError>;

    /// Rows `y0..y0 + rows` as 16-bit interleaved RGB, `rows * width * 3` samples in `out` (which is
    /// cleared first). No dither.
    ///
    /// # Errors
    ///
    /// As [`BandSource::rows_u8`].
    fn rows_u16(&mut self, y0: usize, rows: usize, out: &mut Vec<u16>) -> Result<(), ExportError>;
}

/// An [`ImageRef`] seen as a [`BandSource`]: each call runs [`encode_rows_u8`] or [`encode_rows_u16`]
/// on the CPU. Any order of calls gives the same bytes (the dither uses absolute coordinates).
#[derive(Debug, Clone, Copy)]
pub struct ImageBands<'a> {
    image: ImageRef<'a>,
    space: OutputSpace,
}

impl<'a> ImageBands<'a> {
    /// The rows of `image` encoded for `space`.
    #[must_use]
    pub fn new(image: ImageRef<'a>, space: OutputSpace) -> Self {
        Self { image, space }
    }

    fn check(&self, y0: usize, rows: usize) -> Result<(), ExportError> {
        if y0
            .checked_add(rows)
            .is_none_or(|end| end > self.image.height as usize)
        {
            return Err(ExportError::invalid(format!(
                "rows {y0} to {} are outside the {} rows of the picture",
                y0.saturating_add(rows),
                self.image.height
            )));
        }
        Ok(())
    }
}

impl BandSource for ImageBands<'_> {
    fn width(&self) -> u32 {
        self.image.width
    }

    fn height(&self) -> u32 {
        self.image.height
    }

    fn rows_u8(&mut self, y0: usize, rows: usize, out: &mut Vec<u8>) -> Result<(), ExportError> {
        self.check(y0, rows)?;
        encode_rows_u8(&self.image, self.space, y0, rows, out);
        Ok(())
    }

    fn rows_u16(&mut self, y0: usize, rows: usize, out: &mut Vec<u16>) -> Result<(), ExportError> {
        self.check(y0, rows)?;
        encode_rows_u16(&self.image, self.space, y0, rows, out);
        Ok(())
    }
}

/// Cancel and progress callbacks for one export.
///
/// `cancel` is polled between bands and strips (about every million pixels); returning `true`
/// stops the export with [`ExportError::Cancelled`] and the partial file is deleted. `progress`
/// receives the fraction done, 0 to 1, from any thread.
#[derive(Clone, Copy)]
pub struct ExportHooks<'a> {
    cancel: Option<&'a (dyn Fn() -> bool + Sync)>,
    progress: Option<&'a (dyn Fn(f32) + Sync)>,
    lo: f32,
    hi: f32,
}

impl Default for ExportHooks<'_> {
    fn default() -> Self {
        Self::none()
    }
}

impl std::fmt::Debug for ExportHooks<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExportHooks")
            .field("cancel", &self.cancel.is_some())
            .field("progress", &self.progress.is_some())
            .finish()
    }
}

impl<'a> ExportHooks<'a> {
    /// No cancellation, no progress.
    #[must_use]
    pub const fn none() -> Self {
        Self {
            cancel: None,
            progress: None,
            lo: 0.0,
            hi: 1.0,
        }
    }

    /// Hooks that poll `cancel` and report to `progress`.
    #[must_use]
    pub fn new(
        cancel: Option<&'a (dyn Fn() -> bool + Sync)>,
        progress: Option<&'a (dyn Fn(f32) + Sync)>,
    ) -> Self {
        Self {
            cancel,
            progress,
            lo: 0.0,
            hi: 1.0,
        }
    }

    /// Whether the job was cancelled.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancel.is_some_and(|c| c())
    }

    /// `Err(Cancelled)` when the job was cancelled.
    pub fn check(&self) -> Result<(), ExportError> {
        if self.is_cancelled() {
            Err(ExportError::Cancelled)
        } else {
            Ok(())
        }
    }

    /// Reports `fraction` (0 to 1) of the work these hooks cover.
    pub fn report(&self, fraction: f32) {
        if let Some(p) = self.progress {
            p(self.lo + (self.hi - self.lo) * fraction.clamp(0.0, 1.0));
        }
    }

    /// Hooks for a stage that covers `lo..hi` of this job's progress.
    #[must_use]
    pub fn stage(&self, lo: f32, hi: f32) -> ExportHooks<'a> {
        ExportHooks {
            cancel: self.cancel,
            progress: self.progress,
            lo: self.lo + (self.hi - self.lo) * lo,
            hi: self.lo + (self.hi - self.lo) * hi,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

    #[test]
    fn image_ref_checks_length_and_size() {
        let data = vec![0.0_f32; 2 * 2 * 3];
        assert!(ImageRef::new(2, 2, &data, PixelEncoding::DisplayLinear).is_ok());
        assert!(ImageRef::new(2, 3, &data, PixelEncoding::DisplayLinear).is_err());
        assert!(ImageRef::new(0, 2, &[], PixelEncoding::DisplayLinear).is_err());
    }

    #[test]
    fn band_split_does_not_change_the_bytes() {
        let (w, h) = (37_usize, 29_usize);
        let data: Vec<f32> = (0..w * h * 3)
            .map(|i| ((i * 7919) % 1000) as f32 / 1000.0)
            .collect();
        let img = ImageRef::new(w as u32, h as u32, &data, PixelEncoding::DisplayLinear).unwrap();
        let mut whole = Vec::new();
        encode_rows_u8(&img, OutputSpace::Srgb, 0, h, &mut whole);
        let mut pieces = Vec::new();
        let mut band = Vec::new();
        for (y0, rows) in [(0, 5), (5, 11), (16, 13)] {
            encode_rows_u8(&img, OutputSpace::Srgb, y0, rows, &mut band);
            pieces.extend_from_slice(&band);
        }
        assert_eq!(whole, pieces);
        let mut whole16 = Vec::new();
        encode_rows_u16(&img, OutputSpace::DisplayP3, 0, h, &mut whole16);
        assert_eq!(whole16.len(), w * h * 3);
    }

    #[test]
    fn encoded_input_is_quantised_directly() {
        let data = vec![0.5_f32; 4 * 3];
        let img = ImageRef::new(2, 2, &data, PixelEncoding::DisplayEncoded).unwrap();
        let mut out16 = Vec::new();
        encode_rows_u16(&img, OutputSpace::Srgb, 0, 2, &mut out16);
        assert!(out16.iter().all(|&v| v == 32768));
        let mut out8 = Vec::new();
        encode_rows_u8(&img, OutputSpace::Srgb, 0, 2, &mut out8);
        assert!(out8.iter().all(|&v| v == 127 || v == 128));
    }

    #[test]
    fn hooks_scale_progress_and_poll_cancel() {
        let last = AtomicU32::new(0);
        let stop = AtomicBool::new(false);
        let cancel = || stop.load(Ordering::Relaxed);
        let progress = |f: f32| last.store((f * 1000.0).round() as u32, Ordering::Relaxed);
        let hooks = ExportHooks::new(Some(&cancel), Some(&progress));
        hooks.stage(0.5, 1.0).report(0.5);
        assert_eq!(last.load(Ordering::Relaxed), 750);
        hooks.stage(0.5, 1.0).stage(0.0, 0.5).report(1.0);
        assert_eq!(last.load(Ordering::Relaxed), 750);
        assert!(hooks.check().is_ok());
        stop.store(true, Ordering::Relaxed);
        assert!(matches!(hooks.check(), Err(ExportError::Cancelled)));
        assert!(ExportHooks::none().check().is_ok());
    }
}
