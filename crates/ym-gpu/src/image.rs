//! Images that live on the device: [`GpuImage`] (planar f32) and [`GpuRaw`] (the raw words of one
//! photo), with the host-to-device and device-to-host helpers the tests and the first wiring use
//! (`docs/gpu-plan.md` section 4.3).
//!
//! # Layout
//!
//! A [`GpuImage`] is `planes` planes (1, 3 or 4) of `height` rows of `stride` f32 samples each, one
//! plane after the other, in a single storage buffer. `stride` is `width` rounded up to an even
//! number, like every other image buffer of the crate (`kernels::util::even`); the padding column is
//! zero on upload and kernels never read it. The sample type is **f32 in every storage mode**
//! (`docs/gpu-plan.md` X-3): the storage modes of [`crate::device::StorageMode`] belong to the legacy
//! kernels and their tests.
//!
//! A [`GpuRaw`] is the sensor words of the photo as the decoder produced them, two `u16` per `u32`
//! (little-endian, the first sample in the low half), row after row with no padding: sample
//! `i = y * width + x` is half `i & 1` of word `i >> 1`. An odd `width` therefore shares a word
//! across a row boundary; kernels index by `i`, not by rows of words. One plane of a 61 MP body is
//! 115 MiB, under the 128 MiB binding the plan designs for.
//!
//! Buffers come from the pool ([`crate::pool`]) and go back to it when the image is dropped, so a
//! resident image is an ordinary value that caches can hold behind an `Arc`.

use crate::cpu::{Cfa, Placement, RgbImage};
use crate::device::{GpuContext, GpuError};
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
    /// image whose sample count does not fit a `u32` index (K4 of the plan);
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
    /// None today; the signature leaves room for the length check to become an error.
    pub fn take(&self, done: &Completed, id: ReadId) -> Result<Vec<f32>, GpuError> {
        Ok(self.unpack(done.read(id)))
    }

    /// Decodes the bytes of the image's buffer into dense planar f32 samples, dropping the padding.
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

/// The raw sensor words of one photo, on the device (see the module notes for the layout).
#[derive(Debug)]
pub struct GpuRaw {
    /// Two `u16` per `u32`, `width * height` samples row after row.
    pub words: PooledBuffer,
    /// The quantisation step of a lossy-compressed (cRAW) photo per sample, when it has one. The
    /// representation (a `u8` table index or f16 pairs) is the raw front's to choose.
    pub qstep: Option<PooledBuffer>,
    /// Width in photosites.
    pub width: usize,
    /// Height in photosites.
    pub height: usize,
    /// The 2x2 pattern at photosite (0, 0).
    pub cfa: Cfa,
}

impl GpuRaw {
    /// Bytes of the word buffer for `samples` samples.
    pub fn words_bytes(samples: usize) -> u64 {
        (samples as u64).div_ceil(2) * 4
    }

    /// Uploads the samples of a `width x height` readout once, in chunks of at most
    /// [`CHUNK_BYTES`](crate::stream::CHUNK_BYTES).
    ///
    /// # Errors
    ///
    /// [`GpuError::InvalidInput`] when `samples` is not `width * height` long or the readout is
    /// empty; [`GpuError::BufferTooLarge`] when it does not fit one storage binding;
    /// [`GpuError::OutOfBudget`] and [`GpuError::DeviceLost`] from the pool and the queue.
    pub fn upload(
        ctx: &GpuContext,
        label: &str,
        width: usize,
        height: usize,
        cfa: Cfa,
        samples: &[u16],
    ) -> Result<GpuRaw, GpuError> {
        if width == 0 || height == 0 || samples.len() != width * height {
            return Err(GpuError::InvalidInput(format!(
                "{label}: {} samples for a {width} x {height} readout",
                samples.len()
            )));
        }
        let words = ctx
            .pool()
            .acquire(label, Self::words_bytes(samples.len()), IMAGE_USAGE)?;
        upload_bytes(ctx, &words, 0, bytemuck::cast_slice(samples))?;
        Ok(GpuRaw {
            words,
            qstep: None,
            width,
            height,
            cfa,
        })
    }

    /// Reads the samples back (for tests and for the host fallback).
    ///
    /// # Errors
    ///
    /// The errors of [`read_bytes`].
    pub fn read(&self, ctx: &GpuContext) -> Result<Vec<u16>, GpuError> {
        let n = self.width * self.height;
        let bytes = read_bytes(ctx, &self.words, 0, n as u64 * 2)?;
        Ok(bytes
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect())
    }
}
