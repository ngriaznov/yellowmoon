//! Whole-buffer rendering: [`Frame`] views and the band-parallel pixel loop behind
//! [`ColorChain`]'s `render_*` and `encode_*` methods.
//!
//! # Layouts
//!
//! * [`Layout::Planar`]: all red samples, then all green, then all blue, each plane row-major. This
//!   is the layout of `ym_gpu::cpu::RgbImage`, so its `data` can be passed straight in.
//! * [`Layout::Interleaved`]: `r, g, b, r, g, b, ...` row-major.
//!
//! 8-bit and 16-bit outputs are interleaved RGB (the form image encoders take).
//!
//! # Threads
//!
//! The image is cut into bands of whole rows and the bands are handed out to `threads` scoped
//! threads from a shared queue (no dependency beyond `std`). `threads == 0` means every core the
//! operating system reports, `1` runs on the calling thread. Every pixel depends only on its own
//! input and absolute position, so the output is bit-identical for any thread count and any band
//! size.

use std::fmt;
use std::sync::{Mutex, PoisonError};
use std::thread;

use super::DisplayError;
use super::chain::{ColorChain, View};
use crate::consts::Vec3;

/// Rows per band are chosen so that each thread gets about this many bands (load balancing).
const BANDS_PER_THREAD: usize = 8;
/// A band never has more rows than this.
const MAX_BAND_ROWS: usize = 64;

/// How the three channels of an f32 image are stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Layout {
    /// Three planes (red, green, blue), each `width * height` samples, row-major.
    Planar,
    /// `r, g, b` per pixel, row-major.
    Interleaved,
}

/// A read-only RGB f32 image: dimensions, layout and the samples.
///
/// `Debug` prints the dimensions and the layout, never the samples (a 50 MP frame would be a
/// 600 MB string).
#[derive(Clone, Copy)]
pub struct Frame<'a> {
    width: usize,
    height: usize,
    layout: Layout,
    data: &'a [f32],
}

impl fmt::Debug for Frame<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Frame")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("layout", &self.layout)
            .finish_non_exhaustive()
    }
}

/// `3 * width * height`, or an error when it overflows.
fn sample_count(width: usize, height: usize) -> Result<usize, DisplayError> {
    width
        .checked_mul(height)
        .and_then(|n| n.checked_mul(3))
        .ok_or(DisplayError::TooLarge { width, height })
}

fn check_len(width: usize, height: usize, got: usize) -> Result<(), DisplayError> {
    let expected = sample_count(width, height)?;
    if got == expected {
        Ok(())
    } else {
        Err(DisplayError::BadBuffer {
            got,
            expected,
            width,
            height,
        })
    }
}

impl<'a> Frame<'a> {
    /// A view of `data`, which must hold exactly `3 * width * height` samples.
    ///
    /// # Errors
    ///
    /// [`DisplayError::BadBuffer`] when the length is wrong, [`DisplayError::TooLarge`] on overflow.
    pub fn new(
        width: usize,
        height: usize,
        layout: Layout,
        data: &'a [f32],
    ) -> Result<Self, DisplayError> {
        check_len(width, height, data.len())?;
        Ok(Self {
            width,
            height,
            layout,
            data,
        })
    }

    /// A planar view (the layout of `ym_gpu::cpu::RgbImage::data`).
    ///
    /// # Errors
    ///
    /// As [`Frame::new`].
    pub fn planar(width: usize, height: usize, data: &'a [f32]) -> Result<Self, DisplayError> {
        Self::new(width, height, Layout::Planar, data)
    }

    /// An interleaved view.
    ///
    /// # Errors
    ///
    /// As [`Frame::new`].
    pub fn interleaved(width: usize, height: usize, data: &'a [f32]) -> Result<Self, DisplayError> {
        Self::new(width, height, Layout::Interleaved, data)
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

    /// The sample layout.
    #[must_use]
    pub fn layout(&self) -> Layout {
        self.layout
    }

    /// The pixel at `(x, y)` as f64. `x < width` and `y < height` (the caller loops over the image).
    fn pixel(&self, x: usize, y: usize) -> Vec3 {
        let i = y * self.width + x;
        match self.layout {
            Layout::Planar => {
                let plane = self.width * self.height;
                [
                    f64::from(self.data[i]),
                    f64::from(self.data[plane + i]),
                    f64::from(self.data[2 * plane + i]),
                ]
            }
            Layout::Interleaved => [
                f64::from(self.data[3 * i]),
                f64::from(self.data[3 * i + 1]),
                f64::from(self.data[3 * i + 2]),
            ],
        }
    }
}

/// The destination of one band of rows.
enum BandOut<'a, T> {
    Interleaved(&'a mut [T]),
    Planar([&'a mut [T]; 3]),
}

struct Band<'a, T> {
    first_row: usize,
    rows: usize,
    out: BandOut<'a, T>,
}

/// Cuts `out` into bands of `band_rows` rows (the last may be shorter).
fn split_bands<T>(
    out: &mut [T],
    layout: Layout,
    width: usize,
    height: usize,
    band_rows: usize,
) -> Vec<Band<'_, T>> {
    match layout {
        Layout::Interleaved => out
            .chunks_mut(band_rows * width * 3)
            .enumerate()
            .map(|(i, chunk)| Band {
                first_row: i * band_rows,
                rows: chunk.len() / (width * 3),
                out: BandOut::Interleaved(chunk),
            })
            .collect(),
        Layout::Planar => {
            let plane = width * height;
            let (r, rest) = out.split_at_mut(plane);
            let (g, b) = rest.split_at_mut(plane);
            let size = band_rows * width;
            r.chunks_mut(size)
                .zip(g.chunks_mut(size))
                .zip(b.chunks_mut(size))
                .enumerate()
                .map(|(i, ((r, g), b))| Band {
                    first_row: i * band_rows,
                    rows: r.len() / width,
                    out: BandOut::Planar([r, g, b]),
                })
                .collect()
        }
    }
}

fn resolve_threads(requested: usize, jobs: usize) -> usize {
    let wanted = if requested == 0 {
        thread::available_parallelism().map_or(1, usize::from)
    } else {
        requested
    };
    wanted.clamp(1, jobs.max(1))
}

/// Runs `f(rgb, x, y)` for every pixel of `src` and stores its three results in `out`.
///
/// `(x, y)` are the pixel's coordinates inside `src`.
fn map_pixels<T, F>(
    src: &Frame<'_>,
    out: &mut [T],
    out_layout: Layout,
    threads: usize,
    f: F,
) -> Result<(), DisplayError>
where
    T: Copy + Send,
    F: Fn(Vec3, usize, usize) -> [T; 3] + Sync,
{
    let (width, height) = (src.width, src.height);
    check_len(width, height, out.len())?;
    if width == 0 || height == 0 {
        return Ok(());
    }
    let threads_wanted = resolve_threads(threads, height);
    let band_rows = height
        .div_ceil(threads_wanted * BANDS_PER_THREAD)
        .clamp(1, MAX_BAND_ROWS);
    let bands = split_bands(out, out_layout, width, height, band_rows);
    let run_band = |band: Band<'_, T>| {
        let Band {
            first_row,
            rows,
            out,
        } = band;
        match out {
            BandOut::Interleaved(dst) => {
                for row in 0..rows {
                    let y = first_row + row;
                    for x in 0..width {
                        let v = f(src.pixel(x, y), x, y);
                        let at = 3 * (row * width + x);
                        dst[at..at + 3].copy_from_slice(&v);
                    }
                }
            }
            BandOut::Planar([r, g, b]) => {
                for row in 0..rows {
                    let y = first_row + row;
                    for x in 0..width {
                        let v = f(src.pixel(x, y), x, y);
                        let at = row * width + x;
                        r[at] = v[0];
                        g[at] = v[1];
                        b[at] = v[2];
                    }
                }
            }
        }
    };
    let workers = resolve_threads(threads, bands.len());
    if workers <= 1 {
        bands.into_iter().for_each(run_band);
    } else {
        let queue = Mutex::new(bands.into_iter());
        thread::scope(|scope| {
            for _ in 0..workers {
                scope.spawn(|| {
                    loop {
                        let next = queue.lock().unwrap_or_else(PoisonError::into_inner).next();
                        match next {
                            Some(band) => run_band(band),
                            None => break,
                        }
                    }
                });
            }
        });
    }
    Ok(())
}

fn to_f32(v: Vec3) -> [f32; 3] {
    [v[0] as f32, v[1] as f32, v[2] as f32]
}

impl ColorChain {
    /// Renders camera RGB to encoded display code values in `[0, 1]` (f32, not quantised, not
    /// dithered): the full chain for [`View::Final`], the Before view for [`View::Before`].
    ///
    /// `out` holds `3 * width * height` samples in `out_layout`. `threads` is 0 for all cores.
    ///
    /// # Errors
    ///
    /// [`DisplayError::BadBuffer`] when `out` has the wrong length.
    pub fn render_encoded_f32(
        &self,
        view: View,
        src: &Frame<'_>,
        out_layout: Layout,
        out: &mut [f32],
        threads: usize,
    ) -> Result<(), DisplayError> {
        map_pixels(src, out, out_layout, threads, |rgb, _, _| {
            to_f32(self.apply_view_pixel(view, rgb))
        })
    }

    /// Renders camera RGB to display-linear RGB in the output's primaries (C1 to C12; 1.0 is the
    /// 100 nit white). Resize this in linear light, then call [`ColorChain::encode_display_linear_u8`].
    ///
    /// # Errors
    ///
    /// [`DisplayError::BadBuffer`] when `out` has the wrong length.
    pub fn render_display_linear_f32(
        &self,
        src: &Frame<'_>,
        out_layout: Layout,
        out: &mut [f32],
        threads: usize,
    ) -> Result<(), DisplayError> {
        map_pixels(src, out, out_layout, threads, |rgb, _, _| {
            to_f32(self.apply_to_display_linear(rgb))
        })
    }

    /// Renders the DNG's pixels ([`ColorChain::apply_dng_pixel`]): C1 to C11 decoded back to linear
    /// AWG4 (f32, unclipped), or, when C12 is not our ACES 2 ([`ColorChain::dng_inverts_display`]),
    /// the scene values that ACES 2 turns into the chain's picture.
    ///
    /// # Errors
    ///
    /// [`DisplayError::BadBuffer`] when `out` has the wrong length.
    pub fn render_dng_linear_awg4_f32(
        &self,
        src: &Frame<'_>,
        out_layout: Layout,
        out: &mut [f32],
        threads: usize,
    ) -> Result<(), DisplayError> {
        map_pixels(src, out, out_layout, threads, |rgb, _, _| {
            to_f32(self.apply_dng_pixel(rgb))
        })
    }

    /// Renders camera RGB to 8-bit interleaved RGB (C13 with the chain's dither setting).
    ///
    /// `origin` is the position of `src`'s first pixel in the output image, so the dither pattern
    /// of a tile equals that of the whole image (the position only selects a noise texel, so it
    /// wraps instead of panicking when `origin + size` exceeds `usize`). `out` holds
    /// `3 * width * height` bytes.
    ///
    /// # Errors
    ///
    /// [`DisplayError::BadBuffer`] when `out` has the wrong length.
    pub fn render_u8(
        &self,
        view: View,
        src: &Frame<'_>,
        origin: (usize, usize),
        out: &mut [u8],
        threads: usize,
    ) -> Result<(), DisplayError> {
        map_pixels(src, out, Layout::Interleaved, threads, |rgb, x, y| {
            self.quantize_u8(
                self.apply_view_pixel(view, rgb),
                origin.0.wrapping_add(x),
                origin.1.wrapping_add(y),
            )
        })
    }

    /// Renders camera RGB to 10 to 16-bit interleaved RGB (no dither). `bits` is clamped to 1 to 16.
    ///
    /// # Errors
    ///
    /// [`DisplayError::BadBuffer`] when `out` has the wrong length.
    pub fn render_u16(
        &self,
        view: View,
        bits: u32,
        src: &Frame<'_>,
        out: &mut [u16],
        threads: usize,
    ) -> Result<(), DisplayError> {
        map_pixels(src, out, Layout::Interleaved, threads, |rgb, _, _| {
            self.quantize_u16(self.apply_view_pixel(view, rgb), bits)
        })
    }

    /// C13 on a display-linear image (for example after a linear-light resize): encodes and
    /// quantises to 8-bit interleaved RGB, dithering at absolute pixel `origin + (x, y)`.
    ///
    /// # Errors
    ///
    /// [`DisplayError::BadBuffer`] when `out` has the wrong length.
    pub fn encode_display_linear_u8(
        &self,
        src: &Frame<'_>,
        origin: (usize, usize),
        out: &mut [u8],
        threads: usize,
    ) -> Result<(), DisplayError> {
        map_pixels(src, out, Layout::Interleaved, threads, |rgb, x, y| {
            self.encode_u8(rgb, origin.0.wrapping_add(x), origin.1.wrapping_add(y))
        })
    }

    /// C13 on a display-linear image: encodes and quantises to 10 to 16-bit interleaved RGB.
    ///
    /// # Errors
    ///
    /// [`DisplayError::BadBuffer`] when `out` has the wrong length.
    pub fn encode_display_linear_u16(
        &self,
        bits: u32,
        src: &Frame<'_>,
        out: &mut [u16],
        threads: usize,
    ) -> Result<(), DisplayError> {
        map_pixels(src, out, Layout::Interleaved, threads, |rgb, _, _| {
            self.encode_u16(rgb, bits)
        })
    }
}
