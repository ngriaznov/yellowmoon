//! Images that live on the device: [`GpuImage`] (planar f32), with the host-to-device and
//! device-to-host helpers the tests and the wiring use (`docs/17-gpu.md` section 3). The raw
//! words of a photo are not an image: they live in [`crate::kernels::prep::DeviceRaw`].
//!
//! # Layout
//!
//! A [`GpuImage`] is `planes` planes (1, 3 or 4) of `height` rows of `stride` f32 samples each, one
//! plane after the other, in a single storage buffer. `stride` is `width` rounded up to an even
//! number, like every other image buffer of the crate (`kernels::util::even`); the padding column is
//! zero on upload and kernels never read it. The sample type is **f32 in every storage mode**
//! (`docs/17-gpu.md` section 2): the storage modes of [`crate::device::StorageMode`] belong to the
//! legacy kernels and their tests.
//!
//! Buffers come from the pool ([`crate::pool`]) and go back to it when the image is dropped, so a
//! resident image is an ordinary value that caches can hold behind an `Arc`.

use std::sync::Arc;

use crate::cpu::{Placement, RgbImage};
use crate::device::{GpuContext, GpuError};
use crate::kernels::copy::Piece;
use crate::kernels::util::even;
use crate::pool::PooledBuffer;
use crate::stream::{Completed, PassList, ReadId, read_bytes, upload_bytes};

/// Usage of every image buffer: bound as storage by kernels, copied from for readback, copied to
/// for upload. One mask for all of them lets the pool recycle any image buffer for any other of the
/// same size class.
pub const IMAGE_USAGE: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE
    .union(wgpu::BufferUsages::COPY_SRC)
    .union(wgpu::BufferUsages::COPY_DST);

/// The planar f32 image of a stage's input or output, on the device.
#[derive(Debug)]
pub struct GpuImage {
    /// The storage buffer: `planes * stride * height` f32 samples.
    pub buf: PooledBuffer,
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// Samples per row of one plane: `width` rounded up to an even number.
    pub stride: usize,
    /// Number of planes: 1, 3 or 4.
    pub planes: usize,
    /// Where pixel (0, 0) sits in the frame the image was cut from, and how big that frame is
    /// (kernels that need absolute coordinates read this, as the CPU stages do).
    pub place: Placement,
}

impl GpuImage {
    /// Bytes of the buffer of a `width x height` image with `planes` planes.
    pub fn buffer_bytes(width: usize, height: usize, planes: usize) -> u64 {
        (planes * even(width) * height) as u64 * 4
    }

    /// An image whose content is undefined until a kernel (or an upload) writes it: whatever the
    /// pool had, NaN in poison mode. The pixels the next stage reads must have been written.
    ///
    /// # Errors
    ///
    /// [`GpuError::InvalidInput`] for an empty image, a plane count other than 1, 3 or 4, or an
    /// image whose sample count does not fit a `u32` index (K4 of `docs/17-gpu.md`);
    /// [`GpuError::BufferTooLarge`] when it does not fit one storage binding;
    /// [`GpuError::OutOfBudget`] when the budget cannot hold it.
    pub fn alloc(
        ctx: &GpuContext,
        label: &str,
        width: usize,
        height: usize,
        planes: usize,
        place: Placement,
    ) -> Result<GpuImage, GpuError> {
        if width == 0 || height == 0 {
            return Err(GpuError::InvalidInput(format!(
                "{label}: an image of {width} x {height} pixels is empty"
            )));
        }
        if !matches!(planes, 1 | 3 | 4) {
            return Err(GpuError::InvalidInput(format!(
                "{label}: {planes} planes (an image has 1, 3 or 4)"
            )));
        }
        let stride = even(width);
        let samples = planes as u64 * stride as u64 * height as u64;
        if samples > u64::from(u32::MAX) {
            return Err(GpuError::InvalidInput(format!(
                "{label}: {samples} samples do not fit a 32-bit index"
            )));
        }
        let buf = ctx.pool().acquire(
            label,
            Self::buffer_bytes(width, height, planes),
            IMAGE_USAGE,
        )?;
        Ok(GpuImage {
            buf,
            width,
            height,
            stride,
            planes,
            place,
        })
    }

    /// Samples in one plane, padding included (`stride * height`).
    pub fn plane_len(&self) -> usize {
        self.stride * self.height
    }

    /// Bytes of the image in its buffer.
    pub fn byte_len(&self) -> u64 {
        (self.planes * self.plane_len()) as u64 * 4
    }

    /// Uploads dense planar samples (`planes` planes of `width * height`, row-major, no padding)
    /// as a new image.
    ///
    /// # Errors
    ///
    /// [`GpuError::InvalidInput`] when `data` has the wrong length, plus the errors of
    /// [`GpuImage::alloc`].
    pub fn upload(
        ctx: &GpuContext,
        label: &str,
        width: usize,
        height: usize,
        planes: usize,
        data: &[f32],
        place: Placement,
    ) -> Result<GpuImage, GpuError> {
        if data.len() != planes * width * height {
            return Err(GpuError::InvalidInput(format!(
                "{label}: {} samples for {planes} planes of {width} x {height}",
                data.len()
            )));
        }
        let img = GpuImage::alloc(ctx, label, width, height, planes, place)?;
        let bytes: std::borrow::Cow<'_, [f32]> = if img.stride == width {
            std::borrow::Cow::Borrowed(data)
        } else {
            // Re-pack the rows into the even stride; the padding column is zero.
            let mut padded = vec![0.0f32; planes * img.plane_len()];
            for c in 0..planes {
                for y in 0..height {
                    let src = (c * height + y) * width;
                    let dst = c * img.plane_len() + y * img.stride;
                    padded[dst..dst + width].copy_from_slice(&data[src..src + width]);
                }
            }
            std::borrow::Cow::Owned(padded)
        };
        upload_bytes(ctx, &img.buf, 0, bytemuck::cast_slice(&bytes))?;
        Ok(img)
    }

    /// Uploads a three-plane host image, keeping its placement.
    ///
    /// # Errors
    ///
    /// As [`GpuImage::upload`].
    pub fn upload_rgb(
        ctx: &GpuContext,
        label: &str,
        image: &RgbImage,
    ) -> Result<GpuImage, GpuError> {
        GpuImage::upload(
            ctx,
            label,
            image.width,
            image.height,
            3,
            &image.data,
            image.place,
        )
    }

    /// Reads the image back as dense planar samples (`planes * width * height`, padding removed).
    ///
    /// # Errors
    ///
    /// The errors of [`read_bytes`].
    pub fn read(&self, ctx: &GpuContext) -> Result<Vec<f32>, GpuError> {
        let bytes = read_bytes(ctx, &self.buf, 0, self.byte_len())?;
        Ok(self.unpack(&bytes))
    }

    /// Reads a three-plane image back as a host image, keeping its placement.
    ///
    /// # Errors
    ///
    /// [`GpuError::InvalidInput`] when the image does not have three planes, plus the errors of
    /// [`GpuImage::read`].
    pub fn read_rgb(&self, ctx: &GpuContext) -> Result<RgbImage, GpuError> {
        if self.planes != 3 {
            return Err(GpuError::InvalidInput(format!(
                "an image with {} planes is not an RGB image",
                self.planes
            )));
        }
        let data = self.read(ctx)?;
        Ok(RgbImage {
            width: self.width,
            height: self.height,
            place: self.place,
            data,
        })
    }

    /// Asks `passes` to copy the image to a staging buffer after the passes recorded so far; read
    /// the result with [`GpuImage::unpack`] on the bytes of the returned id once the list has
    /// finished.
    ///
    /// # Errors
    ///
    /// The errors of [`PassList::read`].
    pub fn encode_read(&self, passes: &mut PassList<'_>) -> Result<ReadId, GpuError> {
        passes.read(&self.buf, 0, self.byte_len())
    }

    /// Reads the image out of a finished list, dense planar.
    ///
    /// # Errors
    ///
    /// [`GpuError::InvalidInput`] when the readback `id` is shorter than this image, that is when it
    /// is the readback of another, smaller image.
    ///
    /// # Panics
    ///
    /// When `id` does not belong to `done` (see [`Completed::read`]).
    pub fn take(&self, done: &Completed, id: ReadId) -> Result<Vec<f32>, GpuError> {
        let bytes = done.read(id);
        if (bytes.len() as u64) < self.byte_len() {
            return Err(GpuError::InvalidInput(format!(
                "a readback of {} bytes is too short for a {} x {} image of {} planes ({} bytes)",
                bytes.len(),
                self.width,
                self.height,
                self.planes,
                self.byte_len()
            )));
        }
        Ok(self.unpack(bytes))
    }

    /// Decodes the bytes of the image's buffer into dense planar f32 samples, dropping the padding.
    ///
    /// # Panics
    ///
    /// When `bytes` is shorter than [`GpuImage::byte_len`]; [`GpuImage::take`] checks that first.
    pub fn unpack(&self, bytes: &[u8]) -> Vec<f32> {
        let all: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let mut out = Vec::with_capacity(self.planes * self.width * self.height);
        for c in 0..self.planes {
            for y in 0..self.height {
                let start = c * self.plane_len() + y * self.stride;
                out.extend_from_slice(&all[start..start + self.width]);
            }
        }
        out
    }
}

/// A tall picture held as bands of full-width rows, each band in a buffer of its own, so that no
/// storage binding has to hold the whole picture (`docs/17-gpu.md` K4: at most 128 MiB). The fit
/// view of a 50 MP photo at `k = 2` is 4344 x 2896 pixels, 150 MB in three planes; as bands of a few
/// hundred rows it is a dozen buffers of 5 to 30 MB.
///
/// The bands run from the top without a gap; all have the picture's width, stride and number of
/// planes. The placement of a band is `(0, first row)` in a frame of the picture's size, so kernels
/// that read absolute coordinates (the lens stage) see the picture and not the band.
#[derive(Debug)]
pub struct RowBands {
    /// Width of the picture in pixels.
    pub width: usize,
    /// Height of the picture in rows.
    pub height: usize,
    /// Number of planes (1, 3 or 4).
    pub planes: usize,
    bands: Vec<Arc<GpuImage>>,
}

impl RowBands {
    /// A picture made of `bands`, top to bottom.
    ///
    /// # Errors
    ///
    /// [`GpuError::InvalidInput`] for no band, bands of different widths or plane counts, or bands
    /// that do not follow each other (a band whose placement is not at the row where the one above it
    /// ends).
    pub fn new(bands: Vec<GpuImage>) -> Result<RowBands, GpuError> {
        let Some(first) = bands.first() else {
            return Err(GpuError::InvalidInput(
                "a picture of bands needs at least one band".into(),
            ));
        };
        let (width, planes) = (first.width, first.planes);
        let mut row = 0;
        for b in &bands {
            if (b.width, b.planes) != (width, planes) || b.place.y0 != row || b.place.x0 != 0 {
                return Err(GpuError::InvalidInput(format!(
                    "the bands do not make one picture: a {} x {} x {} band at ({}, {}) follows row {row} of a {width} wide, {planes} plane picture",
                    b.width, b.height, b.planes, b.place.x0, b.place.y0
                )));
            }
            row += b.height;
        }
        Ok(RowBands {
            width,
            height: row,
            planes,
            bands: bands.into_iter().map(Arc::new).collect(),
        })
    }

    /// The bands, top to bottom.
    #[must_use]
    pub fn bands(&self) -> &[Arc<GpuImage>] {
        &self.bands
    }

    /// The bands as pieces of the picture, for [`encode_gather`](crate::kernels::copy::encode_gather).
    #[must_use]
    pub fn pieces(&self) -> Vec<Piece> {
        self.bands
            .iter()
            .map(|b| Piece {
                at: (0, b.place.y0),
                image: Arc::clone(b),
            })
            .collect()
    }

    /// Bytes of all the bands' buffers.
    #[must_use]
    pub fn byte_len(&self) -> u64 {
        self.bands.iter().map(|b| b.byte_len()).sum()
    }

    /// The largest buffer of any band, the figure that must stay under the binding limit.
    #[must_use]
    pub fn max_band_bytes(&self) -> u64 {
        self.bands.iter().map(|b| b.byte_len()).max().unwrap_or(0)
    }

    /// The rows `rows` as one image: a band itself when they lie in one (the window starts at row
    /// `rows.start - band.place.y0` of it, which is the second value), else a copy made into an image
    /// of its own, recorded into `passes` (one copy per plane and band, whole rows). The image keeps
    /// the placement of the rows in the picture, so a kernel that reads absolute coordinates sees
    /// them where they are.
    ///
    /// # Errors
    ///
    /// [`GpuError::InvalidInput`] for rows that are empty or beyond the picture; the pool's errors.
    pub fn window(
        &self,
        passes: &mut PassList<'_>,
        rows: std::ops::Range<usize>,
    ) -> Result<(Arc<GpuImage>, usize), GpuError> {
        if rows.is_empty() || rows.end > self.height {
            return Err(GpuError::InvalidInput(format!(
                "the rows {rows:?} are empty or outside the {} rows of the picture",
                self.height
            )));
        }
        if let Some(b) = self
            .bands
            .iter()
            .find(|b| b.place.y0 <= rows.start && b.place.y0 + b.height >= rows.end)
        {
            return Ok((Arc::clone(b), rows.start - b.place.y0));
        }
        let ctx = passes.context();
        let place = Placement {
            x0: 0,
            y0: rows.start,
            frame_width: self.width,
            frame_height: self.height,
        };
        let dst = GpuImage::alloc(
            ctx,
            "bands window",
            self.width,
            rows.len(),
            self.planes,
            place,
        )?;
        for b in &self.bands {
            let (top, bottom) = (
                b.place.y0.max(rows.start),
                (b.place.y0 + b.height).min(rows.end),
            );
            if top >= bottom {
                continue;
            }
            let len = ((bottom - top) * b.stride * 4) as u64;
            for c in 0..self.planes {
                let from = ((c * b.plane_len() + (top - b.place.y0) * b.stride) * 4) as u64;
                let to = ((c * dst.plane_len() + (top - rows.start) * dst.stride) * 4) as u64;
                passes
                    .encoder()
                    .copy_buffer_to_buffer(&b.buf, from, &dst.buf, to, len);
            }
        }
        Ok((Arc::new(dst), 0))
    }

    /// Reads the picture back as dense planar samples (`planes * width * height`).
    ///
    /// # Errors
    ///
    /// The errors of [`GpuImage::read`].
    pub fn read(&self, ctx: &GpuContext) -> Result<Vec<f32>, GpuError> {
        let n = self.width * self.height;
        let mut out = vec![0.0f32; self.planes * n];
        for b in &self.bands {
            let part = b.read(ctx)?;
            let len = b.width * b.height;
            for c in 0..self.planes {
                let dst = c * n + b.place.y0 * self.width;
                out[dst..dst + len].copy_from_slice(&part[c * len..(c + 1) * len]);
            }
        }
        Ok(out)
    }

    /// Reads a three-plane picture back as a host image of the whole picture.
    ///
    /// # Errors
    ///
    /// [`GpuError::InvalidInput`] when the picture does not have three planes, plus the errors of
    /// [`RowBands::read`].
    pub fn read_rgb(&self, ctx: &GpuContext) -> Result<RgbImage, GpuError> {
        if self.planes != 3 {
            return Err(GpuError::InvalidInput(format!(
                "a picture with {} planes is not an RGB image",
                self.planes
            )));
        }
        Ok(RgbImage {
            width: self.width,
            height: self.height,
            place: Placement::whole(self.width, self.height),
            data: self.read(ctx)?,
        })
    }
}
